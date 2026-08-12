"""
test_recommender_contract.py
==============================
Run this the moment Member 2 hands off their real recommendation function.

It checks that `get_recommendations()` in recommender_interface.py still
matches the contract app.py expects, WITHOUT you having to manually click
through the UI to find out something's broken.

Usage:
    python test_recommender_contract.py

Exit code 0 = everything passes, safe to demo.
Exit code 1 = something's wrong, fix before the presentation.
"""

import sys
import traceback

REQUIRED_KEYS = {
    "track_id",
    "name",
    "artist",
    "genre",
    "year",
    "tags",
    "spotify_preview_url",
    "score",
}

PASSED = []
FAILED = []


def check(name, fn):
    """Run a single test, print PASS/FAIL, and keep going even if it errors."""
    try:
        fn()
        PASSED.append(name)
        print(f"  ✅ {name}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  ❌ {name}\n     -> {e}")
    except Exception as e:
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ❌ {name} (unexpected error)\n     -> {type(e).__name__}: {e}")


def main():
    print("Loading recommender_interface.py ...")
    try:
        from recommender_interface import get_recommendations, get_user_count, get_all_user_ids
    except Exception:
        print("❌ Could not import recommender_interface.py at all.")
        print(traceback.format_exc())
        sys.exit(1)

    try:
        n_users = get_user_count()
        all_ids = get_all_user_ids()
    except Exception:
        print("❌ Could not call get_user_count() / get_all_user_ids().")
        print(traceback.format_exc())
        sys.exit(1)

    valid_user = all_ids[100]  # a real user_id string, not an assumed integer index
    valid_top_n = 5

    print(f"\nRunning contract checks (using user_id={valid_user}, top_n={valid_top_n}) ...\n")

    # --- 1. Basic call works and returns a list -----------------------------
    result_holder = {}

    def basic_call():
        result = get_recommendations(valid_user, valid_top_n)
        result_holder["result"] = result
        assert isinstance(result, list), f"expected list, got {type(result)}"

    check("get_recommendations() returns a list", basic_call)

    result = result_holder.get("result", [])

    # --- 2. List length respects top_n --------------------------------------
    def respects_top_n():
        assert len(result) <= valid_top_n, (
            f"asked for top_n={valid_top_n}, got {len(result)} items back"
        )

    check(f"returns at most top_n ({valid_top_n}) items", respects_top_n)

    # --- 3. Each item is a dict with required keys --------------------------
    def has_required_keys():
        assert len(result) > 0, "result list is empty, can't check keys (is that expected?)"
        for i, item in enumerate(result):
            assert isinstance(item, dict), f"item {i} is not a dict, got {type(item)}"
            missing = REQUIRED_KEYS - set(item.keys())
            assert not missing, f"item {i} is missing keys: {missing}"

    check("each recommendation has all required keys", has_required_keys)

    # --- 4. Field types are sane ---------------------------------------------
    def field_types_ok():
        for i, item in enumerate(result):
            assert isinstance(item["name"], str), f"item {i}: 'name' should be a string"
            assert isinstance(item["artist"], str), f"item {i}: 'artist' should be a string"
            assert isinstance(item["year"], int), f"item {i}: 'year' should be an int"
            assert isinstance(item["score"], (int, float)), f"item {i}: 'score' should be numeric"

    check("field types match what the UI expects", field_types_ok)

    # --- 5. No duplicate songs in one result set -----------------------------
    def no_duplicates():
        track_ids = [item["track_id"] for item in result]
        assert len(track_ids) == len(set(track_ids)), "duplicate track_ids in one recommendation list"

    check("no duplicate songs in a single result", no_duplicates)

    # --- 6. Invalid user_id raises a clear error, not a crash ---------------
    def invalid_user_handled():
        fake_user_id = "this_user_id_does_not_exist_12345"
        try:
            get_recommendations(fake_user_id, valid_top_n)
        except ValueError:
            return  # correct behavior
        except Exception as e:
            raise AssertionError(
                f"expected ValueError for invalid user_id, got {type(e).__name__}: {e}"
            )
        raise AssertionError("expected an error for an unknown user_id, but no error was raised")

    check("invalid user_id raises ValueError, not a silent failure", invalid_user_handled)

    # --- 7. top_n=0 or negative is handled ------------------------------------
    def invalid_top_n_handled():
        try:
            get_recommendations(valid_user, 0)
        except ValueError:
            return
        except Exception as e:
            raise AssertionError(
                f"expected ValueError for top_n=0, got {type(e).__name__}: {e}"
            )
        raise AssertionError("expected an error for top_n=0, but no error was raised")

    check("top_n=0 raises ValueError, not a silent failure", invalid_top_n_handled)

    # --- 8. Requesting more than exists doesn't crash ------------------------
    def large_top_n_handled():
        big_result = get_recommendations(valid_user, 10_000)
        assert isinstance(big_result, list), "should still return a list, just shorter"

    check("very large top_n doesn't crash (just returns fewer/what's available)", large_top_n_handled)

    # --- 9. last user in the array works (not just an arbitrary early one) --
    def edge_user_handled():
        get_recommendations(all_ids[-1], valid_top_n)

    check("last user_id in user_ids.npy doesn't crash", edge_user_handled)

    # --- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    print("=" * 60)

    if FAILED:
        print("\nFailed checks:")
        for name, reason in FAILED:
            print(f"  - {name}: {reason}")
        print(
            "\n⚠️  Fix these before demoing — app.py expects this exact contract.\n"
            "   See the docstring at the top of recommender_interface.py."
        )
        sys.exit(1)
    else:
        print("\n✅ All good — safe to plug into the Streamlit app as-is.")
        sys.exit(0)


if __name__ == "__main__":
    main()
