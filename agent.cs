using System.ClientModel;
using System.IO.Compression;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;
using OpenAI;
using OpenAI.Chat;

namespace AgentDot;

record StoredMessage(string Role, string Content);

sealed class Agent {
    const string Endpoint = "https://api.deepseek.com/v1";
    const string SummaryModel = "deepseek-chat";   // always the cheap model for summaries
    const int TokenBudget = 3000;  // compact history once it exceeds this (estimated tokens)
    const int KeepRecent = 6;      // messages kept verbatim after compaction
    const byte Magic = 0x01;       // file header: 0x01 + Brotli(JSON). Never a valid JSON start.

    static readonly JsonSerializerOptions JsonOptions = new() {
        // Keep accents / non-ASCII as raw UTF-8 instead of \uXXXX escapes (much smaller for Spanish).
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
    static readonly Regex BlankRuns = new(@"\n{3,}", RegexOptions.Compiled);

    readonly ChatClient client;
    readonly ChatClient summarizer;
    readonly List<ChatMessage> history = [];
    string memoryPath = DefaultMemoryPath();

    public Agent(string apiKey, string model) {
        client = MakeClient(apiKey, model);
        summarizer = model == SummaryModel ? client : MakeClient(apiKey, SummaryModel);
    }

    static ChatClient MakeClient(string apiKey, string model) => new(
        model: model,
        credential: new ApiKeyCredential(apiKey),
        options: new OpenAIClientOptions { Endpoint = new Uri(Endpoint) });

    static string DefaultMemoryPath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".ermai", "memory.bin");

    // ---------- Memory ----------

    public void LoadMemory(string? path) {
        if (!string.IsNullOrWhiteSpace(path)) memoryPath = path;

        // Migrate: if the new file doesn't exist yet, fall back to a legacy memory.json next to it.
        var source = memoryPath;
        if (!File.Exists(source)) {
            var legacy = Path.ChangeExtension(memoryPath, ".json");
            if (!File.Exists(legacy)) return;
            source = legacy;
        }

        try {
            var rows = ReadRows(File.ReadAllBytes(source));
            history.Clear();
            history.AddRange(rows.Where(r => r.Length == 2).Select(r => ToMessage(r[0], r[1])));
        }
        catch (Exception ex) when (ex is JsonException or InvalidDataException or IOException or UnauthorizedAccessException) {
            var backup = source + ".bak";
            try { File.Copy(source, backup, overwrite: true); } catch { /* best effort */ }
            Console.Error.WriteLine($"[warning] Could not read '{source}' ({ex.Message}); starting fresh. Backup: {backup}");
        }
    }

    static List<string[]> ReadRows(byte[] bytes) {
        if (bytes.Length > 0 && bytes[0] == Magic) {
            using var ms = new MemoryStream(bytes, 1, bytes.Length - 1);
            using var br = new BrotliStream(ms, CompressionMode.Decompress);
            return JsonSerializer.Deserialize<List<string[]>>(br) ?? [];
        }

        // Legacy format: [{"Role": "...", "Content": "..."}]
        var old = JsonSerializer.Deserialize<List<StoredMessage>>(bytes) ?? [];
        return old.Select(m => new[] { m.Role switch { "system" => "s", "assistant" => "a", _ => "u" }, m.Content }).ToList();
    }

    public void SaveMemory() {
        try {
            var dir = Path.GetDirectoryName(memoryPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            // Minimal JSON: [["u","hi"],["a","hello"]] -> no indentation, no property names.
            var rows = history.Select(m => new[] { RoleOf(m), TextOf(m) });
            var json = JsonSerializer.SerializeToUtf8Bytes(rows, JsonOptions);

            using var ms = new MemoryStream();
            ms.WriteByte(Magic);
            using (var br = new BrotliStream(ms, CompressionLevel.SmallestSize, leaveOpen: true))
                br.Write(json);

            // Write to a temp file and swap, so a crash can't leave a half-written file.
            var tmp = memoryPath + ".tmp";
            File.WriteAllBytes(tmp, ms.ToArray());
            File.Move(tmp, memoryPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) {
            Console.Error.WriteLine($"[warning] Could not save memory: {ex.Message}");
        }
    }

    public void ClearMemory() {
        history.Clear();
        SaveMemory();
    }

    static ChatMessage ToMessage(string role, string content) => role switch {
        "s" => ChatMessage.CreateSystemMessage(content),
        "a" => ChatMessage.CreateAssistantMessage(content),
        _ => ChatMessage.CreateUserMessage(content),
    };

    static string RoleOf(ChatMessage m) => m switch {
        SystemChatMessage => "s",
        AssistantChatMessage => "a",
        _ => "u",
    };

    static string TextOf(ChatMessage m) => string.Concat(m.Content.Select(c => c.Text ?? ""));

    // Light normalization before storing: fewer tokens, content unchanged.
    static string Normalize(string s) => BlankRuns.Replace(s.Trim(), "\n\n");

    // ---------- Token budget ----------

    // Rough estimate (~4 chars per token). Good enough to decide when to compact.
    static int EstimateTokens(string s) => (s.Length + 3) / 4;

    int HistoryTokens() => history.Sum(m => EstimateTokens(TextOf(m)) + 4);

    /// <summary>
    /// When history exceeds the budget, replaces everything except the last few messages
    /// with one short summary. Between compactions history is append-only, so the request
    /// prefix stays identical and DeepSeek's context cache can keep hitting.
    /// </summary>
    async Task CompactAsync(CancellationToken ct) {
        if (HistoryTokens() <= TokenBudget) return;

        // Keep the last KeepRecent messages, starting on a user message.
        var keepFrom = Math.Max(0, history.Count - KeepRecent);
        while (keepFrom < history.Count && history[keepFrom] is not UserChatMessage) keepFrom++;
        if (keepFrom == 0) return;

        var old = history.Take(keepFrom).ToList();
        var recent = history.Skip(keepFrom).ToList();

        try {
            var transcript = string.Join("\n", old.Select(m => $"{RoleOf(m)}: {TextOf(m)}"));
            var request = new List<ChatMessage>
            {
                ChatMessage.CreateSystemMessage(
                    "Compress this conversation into a terse note for your own future context. " +
                    "Keep: facts, decisions, names, numbers, code identifiers, user preferences, open tasks. " +
                    "Drop: pleasantries, repetition, anything derivable. " +
                    "Max 150 words, plain text, same language as the conversation. " +
                    "Roles: u=user, a=assistant, s=earlier summary."),
                ChatMessage.CreateUserMessage(transcript),
            };

            ChatCompletion result = (await summarizer.CompleteChatAsync(request, cancellationToken: ct)).Value;
            var summary = result.Content.Count > 0 ? result.Content[0].Text?.Trim() : null;
            if (string.IsNullOrEmpty(summary)) throw new InvalidOperationException("empty summary");

            history.Clear();
            history.Add(ChatMessage.CreateSystemMessage("Earlier conversation summary: " + summary));
            history.AddRange(recent);
        }
        catch (Exception ex) when (ex is ClientResultException or HttpRequestException or InvalidOperationException) {
            // Fallback: just drop the oldest messages until we fit.
            while (history.Count > KeepRecent && HistoryTokens() > TokenBudget)
                history.RemoveAt(0);
            while (history.Count > 0 && history[0] is AssistantChatMessage)
                history.RemoveAt(0);
        }
    }

    // ---------- Chat ----------

    /// <summary>
    /// Sends a request and streams the reply to <paramref name="onToken"/> as it arrives.
    /// History is only updated if the whole exchange succeeds.
    /// </summary>
    public async Task<string> AskAsync(string request, Action<string> onToken, CancellationToken ct = default) {
        await CompactAsync(ct);

        request = request.Trim();
        var messages = new List<ChatMessage>(history) { ChatMessage.CreateUserMessage(request) };
        var reply = new StringBuilder();

        await foreach (var update in client.CompleteChatStreamingAsync(messages, options: null, cancellationToken: ct)) {
            foreach (var part in update.ContentUpdate) {
                if (string.IsNullOrEmpty(part.Text)) continue;
                reply.Append(part.Text);
                onToken(part.Text);
            }
        }

        var text = reply.ToString();
        history.Add(ChatMessage.CreateUserMessage(request));
        history.Add(ChatMessage.CreateAssistantMessage(Normalize(text)));
        return text;
    }
}
