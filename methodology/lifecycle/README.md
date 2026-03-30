# Lifecycle

The lifecycle is the core operating system of this methodology. Each phase has a defined purpose, explicit inputs and outputs, and clear exit criteria. The sequence is not ceremonial — each phase prevents a class of failures that the next phase cannot catch.

```
Exploration → Architecture → Critique → Implementation → Review → Evaluation → Refactor
```

## Files

| File | Purpose |
|------|---------|
| [exploration.md](exploration.md) | Validate the hypothesis; decide whether to build |
| [architecture.md](architecture.md) | Specify the system before any code is written |
| [critique.md](critique.md) | Find every way the design can fail, before implementation |
| [implementation.md](implementation.md) | Build to the specification, with tests alongside code |
| [review.md](review.md) | Verify the implementation matches the specification |
| [evaluation.md](evaluation.md) | Measure system behavior under realistic conditions |
| [refactor.md](refactor.md) | Improve what is causing friction or fragility, incrementally |

## The Critical Phase

Start with [critique.md](critique.md). It is the phase most engineers skip, and the one that prevents the most expensive mistakes.
