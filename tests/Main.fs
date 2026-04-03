module OBrienMcp.Tests.Main

open Expecto

[<EntryPoint>]
let main argv =
    runTestsWithCLIArgs [] argv (testList "O-Brien-mcp" [
        DecayTests.tests
        IntegrationTests.tests
        ToolsTests.tests
    ])
