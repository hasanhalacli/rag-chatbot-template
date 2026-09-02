## What & why

<!-- Link the plan file or ticket. -->

## AI involvement

- [ ] Substantially agent-written
- [ ] Agent-assisted
- [ ] Manual

## What I personally verified

<!-- Not what the tests cover — what YOU ran, watched, and confirmed.
     Example: "Ran the 101-request loop locally, saw the 429 + Retry-After." -->

## Checklist (agent-written changes)

- [ ] Diff read in blast-radius order (deps → config → auth → code → tests)
- [ ] Every touched file is in the plan, or justified here
- [ ] No new/loosened dependencies without approval
- [ ] Tests assert behavior and go red when the feature is broken
