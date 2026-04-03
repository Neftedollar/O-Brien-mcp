module OBrienMcp.Tests.DecayTests

open System
open Expecto
open OBrienMcp.Domain
open OBrienMcp.Db

// All tests assume MEMORY_DECAY_HALF_LIFE is unset (default = 30.0 days).

let private mem (createdAgo: TimeSpan) (accessedAgo: TimeSpan) accessCount tags =
    let now = DateTime.UtcNow
    { Id             = Guid.NewGuid()
      Content        = "test"
      Category       = "test"
      Tags           = tags
      AccessCount    = accessCount
      LastAccessedAt = now - accessedAgo
      CreatedAt      = now - createdAgo
      UpdatedAt      = now - createdAgo }

let private days d = TimeSpan.FromDays(float d)

let tests =
    testList "calculateStrength" [

        test "evergreen tag → 1.0 regardless of age" {
            let m = mem (days 365) (days 365) 0 [|"evergreen"|]
            Expect.equal (calculateStrength m) 1.0 "evergreen always 1.0"
        }

        test "never-forget tag → 1.0 regardless of age" {
            let m = mem (days 365) (days 365) 0 [|"never-forget"|]
            Expect.equal (calculateStrength m) 1.0 "never-forget always 1.0"
        }

        test "brand-new memory → strength near 1.0" {
            // age ≈ 0 → 0.5^0 ≈ 1.0
            let m = mem TimeSpan.Zero TimeSpan.Zero 0 [||]
            Expect.isGreaterThan (calculateStrength m) 0.99 "age=0 → ~1.0"
        }

        test "memory aged exactly half-life → strength near 0.5" {
            // age=30d, last accessed 31d ago → recencyBoost=1.0, accessBoost=1.0
            // strength = 0.5^(30 / (30 × 1.0 × 1.0)) = 0.5
            let m = mem (days 30) (days 31) 0 [||]
            let s = calculateStrength m
            Expect.isGreaterThan s 0.49 "30-day-old memory should be ≥ 0.49"
            Expect.isLessThan    s 0.51 "30-day-old memory should be ≤ 0.51"
        }

        test "two-half-lives old → strength near 0.25" {
            // age=60d, no boosts → 0.5^2 = 0.25
            let m = mem (days 60) (days 61) 0 [||]
            let s = calculateStrength m
            Expect.isGreaterThan s 0.23 "60-day-old memory ≥ 0.23"
            Expect.isLessThan    s 0.27 "60-day-old memory ≤ 0.27"
        }

        test "older memories are weaker (all else equal)" {
            let young = mem (days 10) (days 31) 0 [||]
            let old   = mem (days 60) (days 61) 0 [||]
            Expect.isLessThan (calculateStrength old) (calculateStrength young) "older → weaker"
        }

        test "strength always in (0, 1]" {
            [ 0; 10; 30; 90; 365; 1000 ]
            |> List.iter (fun d ->
                let m = mem (days d) (days (d + 1)) 0 [||]
                let s = calculateStrength m
                Expect.isGreaterThan     s 0.0 $"strength > 0 at {d} days"
                Expect.isLessThanOrEqual s 1.0 $"strength ≤ 1 at {d} days")
        }

        test "recent access (< 7 days) boosts strength via recencyBoost=1.5" {
            // same age, both have 0 accesses; difference is when last accessed
            let dormant = mem (days 30) (days 31) 0 [||]  // recencyBoost = 1.0
            let recent  = mem (days 30) (days 1)  0 [||]  // recencyBoost = 1.5
            Expect.isGreaterThan (calculateStrength recent) (calculateStrength dormant)
                "recently accessed memory should be stronger"
        }

        test "medium recency (7–30 days) gives recencyBoost=1.2" {
            let dormant = mem (days 30) (days 31) 0 [||]  // recencyBoost = 1.0
            let medium  = mem (days 30) (days 10) 0 [||]  // recencyBoost = 1.2
            let recent  = mem (days 30) (days 1)  0 [||]  // recencyBoost = 1.5
            let sDormant = calculateStrength dormant
            let sMedium  = calculateStrength medium
            let sRecent  = calculateStrength recent
            Expect.isGreaterThan sMedium sDormant "medium recency > dormant"
            Expect.isGreaterThan sRecent sMedium  "very recent > medium recency"
        }

        test "higher access count boosts strength via accessBoost" {
            // accessBoost = 1.0 + min(count, 20) / 20
            let cold = mem (days 30) (days 31)  0 [||]   // accessBoost = 1.0
            let warm = mem (days 30) (days 31) 10 [||]   // accessBoost = 1.5
            let hot  = mem (days 30) (days 31) 20 [||]   // accessBoost = 2.0
            let sCold = calculateStrength cold
            let sWarm = calculateStrength warm
            let sHot  = calculateStrength hot
            Expect.isGreaterThan sWarm sCold "10 accesses > 0 accesses"
            Expect.isGreaterThan sHot  sWarm "20 accesses > 10 accesses"
        }

        test "access count saturates at 20" {
            let cap  = mem (days 30) (days 31) 20  [||]
            let over = mem (days 30) (days 31) 999 [||]
            let diff = abs (calculateStrength cap - calculateStrength over)
            Expect.isLessThan diff 1e-9 "access count beyond 20 has no additional effect"
        }

        test "multiple tags: any evergreen tag triggers 1.0" {
            let m = mem (days 365) (days 365) 0 [|"important"; "evergreen"; "archived"|]
            Expect.equal (calculateStrength m) 1.0 "any evergreen tag → 1.0"
        }

        // This test only runs when MEMORY_DECAY_HALF_LIFE=0 is set before process start.
        // The module-level `halfLife` is initialized once on first load, so the env var
        // must be present at startup. Run with:
        //   MEMORY_DECAY_HALF_LIFE=0 dotnet run --project tests -- --filter "halfLife=0"
        test "halfLife=0 env var → strength always 1.0 for any non-evergreen memory" {
            let envVal =
                System.Environment.GetEnvironmentVariable("MEMORY_DECAY_HALF_LIFE")
                |> Option.ofObj
                |> Option.bind (fun s -> match System.Double.TryParse(s) with true, v -> Some v | _ -> None)
                |> Option.defaultValue 30.0
            if envVal <> 0.0 then
                Tests.skiptest "set MEMORY_DECAY_HALF_LIFE=0 before process start to run this test"
            let aged = mem (days 365) (days 365) 0 [||]
            Expect.equal (calculateStrength aged) 1.0 "halfLife=0 disables decay entirely"
        }

    ]
