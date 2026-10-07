"""Neural Forge: model introspection, interventions and checkpoints.

Layering (each module only imports from the ones above it)::

    mlp            functional network + forward trace (pure PyTorch)
    interventions  non-destructive what-if overlay
    checkpoints    bounded snapshot store
    session        a model + dataset + training + checkpoints
    schema         wire contract (pydantic models only)
    introspect     session -> schema payloads (the only torch <-> UI bridge)
    api            FastAPI router

See docs/ARCHITECTURE.md for the full picture.
"""
