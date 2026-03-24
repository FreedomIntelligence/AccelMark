## Summary

<!-- What does this PR do? One paragraph. -->

## Type of change

- [ ] New platform support
- [ ] Bug fix (runner, validator, leaderboard, or tooling)
- [ ] Suite definition change
- [ ] Schema change
- [ ] Leaderboard / UI improvement
- [ ] Documentation
- [ ] Other: <!-- describe -->

## Testing

<!-- How did you test this? What did you run? -->

```bash
# Commands used to verify
```

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] My change does not break existing `result.json` files (or I have explained the migration path)
- [ ] If adding a new platform: runner inherits from `BenchmarkRunner`, produces valid `result.json`, includes a reference result
- [ ] If changing the schema: `validate_submission.py` updated and all existing results still validate
- [ ] If changing the leaderboard generator: `leaderboard/generate.py` produces correct output on existing results
- [ ] I have updated relevant documentation

## Related issues

<!-- Closes #<number> -->
