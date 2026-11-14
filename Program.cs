using System.ClientModel;
using System.CommandLine;

namespace AgentDot;

public static class Program {
    const string ApiKeyVar = "COPILOT_PROVIDER_API_KEY";

    static readonly Option<string> levelOption = new("--level", "-l") {
        Description = "Reasoning level: low (deepseek-chat) or high (deepseek-reasoner)",
        DefaultValueFactory = _ => "low",
    };
    static readonly Option<bool> interactiveOption = new("--interactive", "-i") {
        Description = "Start an interactive chat session",
    };
    static readonly Option<string> memoryOption = new("--memory", "-m") {
        Description = "Path to the memory file (compressed). Defaults to ~/.ermai/memory.bin",
    };
    static readonly Argument<string> userInput = new("input") {
        Description = "User request",
        Arity = ArgumentArity.ZeroOrOne,
    };

    public static async Task<int> Main(string[] args) {
        levelOption.AcceptOnlyFromAmong("low", "high");

        var rootCommand = new RootCommand("Command line AI agent");
        rootCommand.Options.Add(levelOption);
        rootCommand.Options.Add(interactiveOption);
        rootCommand.Options.Add(memoryOption);
        rootCommand.Arguments.Add(userInput);

        rootCommand.SetAction(RunAsync);

        return await rootCommand.Parse(args).InvokeAsync();
    }

    static async Task<int> RunAsync(ParseResult parseResult, CancellationToken ct) {
        var apiKey = Environment.GetEnvironmentVariable(ApiKeyVar);
        if (string.IsNullOrWhiteSpace(apiKey)) {
            Console.Error.WriteLine($"Error: set the {ApiKeyVar} environment variable.");
            return 1;
        }

        var model = parseResult.GetValue(levelOption) == "high" ? "deepseek-reasoner" : "deepseek-chat";
        var agent = new Agent(apiKey, model);
        agent.LoadMemory(parseResult.GetValue(memoryOption));

        var input = parseResult.GetValue(userInput);
        try {
            if (parseResult.GetValue(interactiveOption) || string.IsNullOrWhiteSpace(input))
                await RunInteractiveAsync(agent, ct);
            else
                await AskAndPrintAsync(agent, input, ct);
            return 0;
        }
        catch (OperationCanceledException) {
            Console.Error.WriteLine("\n[cancelled]");
            return 130;
        }
        catch (ClientResultException ex) {
            Console.Error.WriteLine($"\n[API error {ex.Status}] {ex.Message}");
            return 1;
        }
        catch (HttpRequestException ex) {
            Console.Error.WriteLine($"\n[network error] {ex.Message}");
            return 1;
        }
    }

    static async Task AskAndPrintAsync(Agent agent, string input, CancellationToken ct) {
        await agent.AskAsync(input, Console.Write, ct);
        Console.WriteLine();
        agent.SaveMemory();
    }

    static async Task RunInteractiveAsync(Agent agent, CancellationToken ct) {
        Console.WriteLine("Interactive mode. Commands: /clear (reset memory), /exit (save & quit).");

        while (!ct.IsCancellationRequested) {
            Console.Write("> ");
            var line = Console.ReadLine()?.Trim();
            if (line is null) break;          // EOF (Ctrl+D / Ctrl+Z)
            if (line.Length == 0) continue;

            switch (line.ToLowerInvariant()) {
                case "/exit":
                case "/quit":
                    agent.SaveMemory();
                    return;

                case "/clear":
                    agent.ClearMemory();
                    Console.WriteLine("[memory cleared]");
                    break;

                default:
                    try {
                        await AskAndPrintAsync(agent, line, ct);
                    }
                    catch (Exception ex) when (ex is ClientResultException or HttpRequestException) {
                        // A failed request shouldn't kill the session; history was left untouched.
                        Console.Error.WriteLine($"\n[error] {ex.Message}");
                    }
                    break;
            }
        }

        agent.SaveMemory();
    }
}