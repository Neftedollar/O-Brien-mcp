module OBrienMcp.Program

open System
open FsMcp.Server

let private buildServer () =
    let base_ = mcpServer {
        name "O-Brien Memory"
        version "1.0.0"
        useStdio
    }
    { base_ with Tools = Tools.all }

[<EntryPoint>]
let main _ =
    // Redirect .NET logs to stderr so stdout stays clean for MCP protocol
    Console.Error.WriteLine("[obrien-mcp] starting up…")
    Console.Error.WriteLine($"[obrien-mcp] embedding model : {Embeddings.embeddingModel}")
    Console.Error.WriteLine($"[obrien-mcp] ollama url      : {Embeddings.ollamaUrl}")

    task {
        try
            do! Db.initSchema()
            Console.Error.WriteLine("[obrien-mcp] database schema ready")
        with ex ->
            Console.Error.WriteLine($"[obrien-mcp] WARNING: schema init failed: {ex.Message}")

        let server = buildServer()
        do! Server.run server
    }
    |> fun t -> t.GetAwaiter().GetResult()

    0
