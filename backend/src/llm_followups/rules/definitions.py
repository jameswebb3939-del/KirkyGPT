from __future__ import annotations

from .models import AnswerBranch, ConversationRule, RuleStep


DOCKER_RULE = ConversationRule(
    id="docker",
    keywords=("docker", "dockerfile", "container", "containers", "compose"),
    steps=(
        RuleStep(
            id="docker_environment",
            question="Are you using Docker for local development, deployment, or both?",
            branches=(
                AnswerBranch(
                    ("local", "development", "dev"),
                    "For local development, publish only the ports you need, use bind mounts where live updates help, and keep development-only settings separate from production.",
                ),
                AnswerBranch(
                    ("deployment", "deploy", "production", "prod"),
                    "For deployment, prefer immutable images, explicit configuration, health checks, and persistent storage only for state that must survive container replacement.",
                ),
                AnswerBranch(
                    ("both",),
                    "For both, share the same base image and core service definitions, then layer environment-specific settings on top.",
                ),
            ),
            default_response="Keep local and deployment configuration separate so development choices do not accidentally become production defaults.",
        ),
        RuleStep(
            id="docker_area",
            question="Do you need help with Dockerfiles, containers, or Docker Compose?",
            branches=(
                AnswerBranch(
                    ("dockerfile", "dockerfiles", "image", "images"),
                    "For Dockerfiles, keep layers reproducible, copy dependency metadata before application code when useful for caching, and use a small runtime image where practical.",
                ),
                AnswerBranch(
                    ("compose",),
                    "For Docker Compose, give each service one clear role, use service names for container-to-container DNS, and publish ports only when the host needs access.",
                ),
                AnswerBranch(
                    ("container", "containers"),
                    "For containers, keep durable state outside the container, expose health signals, and make startup configuration explicit.",
                ),
            ),
            default_response="Treat Dockerfiles as image definitions, containers as running instances, and Compose as orchestration for a multi-service application.",
        ),
    ),
)


PYTEST_RULE = ConversationRule(
    id="pytest",
    keywords=("pytest", "unit test", "unit tests", "integration test", "integration tests", "failing test"),
    steps=(
        RuleStep(
            id="pytest_goal",
            question="Are you learning pytest basics or debugging a failing test?",
            branches=(
                AnswerBranch(
                    ("basic", "basics", "learn", "learning"),
                    "For pytest basics, start with small test functions using plain assertions, then introduce fixtures only when setup is genuinely shared.",
                ),
                AnswerBranch(
                    ("debug", "debugging", "fail", "failing", "failure"),
                    "For a failing test, isolate the first incorrect assumption: check the assertion, fixture inputs, mocked boundaries, and the smallest reproducible test.",
                ),
            ),
            default_response="Keep the test focused on one observable behaviour so it is clear whether the problem comes from setup, execution, or the assertion.",
        ),
        RuleStep(
            id="pytest_area",
            question="Is the main issue assertions, fixtures, mocks, async tests, or test structure?",
            branches=(
                AnswerBranch(("assert", "assertion", "assertions"), "For assertions, compare the behaviour that matters rather than incidental implementation details."),
                AnswerBranch(("fixture", "fixtures"), "For fixtures, keep scope as narrow as practical and avoid hiding too much behaviour inside fixture setup."),
                AnswerBranch(("mock", "mocks", "mocking", "monkeypatch"), "For mocks, replace only external or expensive boundaries and test the behaviour of your code rather than the mocked implementation."),
                AnswerBranch(("async", "asyncio", "pytest-asyncio"), "For async tests, keep event-loop ownership consistent and await the real asynchronous boundary."),
                AnswerBranch(("structure", "layout", "organization"), "For test structure, separate unit and integration tests by purpose and keep names aligned with the behaviour under test."),
            ),
            default_response="Choose the smallest test layer that reproduces the problem and keep unrelated setup out of it.",
        ),
    ),
)


FASTAPI_RULE = ConversationRule(
    id="fastapi",
    keywords=("fastapi", "api endpoint", "api endpoints", "rest api"),
    steps=(
        RuleStep(
            id="fastapi_change",
            question="Are you creating a new FastAPI service or modifying an existing one?",
            branches=(
                AnswerBranch(("new", "create", "creating", "build", "building"), "For a new FastAPI service, define request and response contracts first, then separate transport, business logic, and persistence."),
                AnswerBranch(("existing", "modify", "modifying", "change", "changing"), "For an existing service, preserve the current API contract unless the change intentionally versions it, and add regression tests around the changed behaviour."),
            ),
            default_response="Keep FastAPI responsible for the HTTP boundary and move domain behaviour into testable services.",
        ),
        RuleStep(
            id="fastapi_area",
            question="Is the main area routes, request validation, response models, or error handling?",
            branches=(
                AnswerBranch(("route", "routes", "endpoint", "endpoints"), "For routes, keep handlers thin and make status codes and endpoint responsibilities explicit."),
                AnswerBranch(("validation", "request", "pydantic"), "For request validation, reject invalid data at the Pydantic boundary before it reaches application services."),
                AnswerBranch(("response", "responses", "schema"), "For response models, expose a stable API schema rather than leaking persistence objects directly."),
                AnswerBranch(("error", "errors", "exception", "exceptions"), "For error handling, translate domain failures into consistent HTTP responses at the API boundary."),
            ),
            default_response="Keep the HTTP contract explicit so routes, validation, responses, and errors can be tested independently.",
        ),
        RuleStep(
            id="fastapi_runtime",
            question="Does the implementation mainly involve async I/O, dependency injection, or API testing?",
            branches=(
                AnswerBranch(("async", "asyncio", "i/o", "io"), "For async I/O, await network and database operations without blocking the event loop."),
                AnswerBranch(("dependency", "dependencies", "injection", "di"), "For dependency injection, inject infrastructure at clear boundaries so tests can replace it cleanly."),
                AnswerBranch(("test", "testing", "testclient", "httpx"), "For API testing, cover validation, status codes, response bodies, and important failure paths through the public HTTP boundary."),
            ),
            default_response="Keep runtime concerns isolated behind interfaces so the API layer remains straightforward to test.",
        ),
    ),
)


DATABASE_RULE = ConversationRule(
    id="database",
    keywords=("sqlalchemy", "database", "sqlite", "postgres", "postgresql", "mysql", "repository", "unit of work"),
    steps=(
        RuleStep(
            id="database_style",
            question="Are you using synchronous or asynchronous database access?",
            branches=(
                AnswerBranch(("async", "asynchronous"), "For asynchronous access, keep one clear async session lifecycle per unit of work and await database I/O through the service boundary."),
                AnswerBranch(("sync", "synchronous"), "For synchronous access, keep session ownership explicit and avoid sharing mutable sessions across independent requests."),
            ),
            default_response="Make session and transaction ownership explicit rather than letting it leak across layers.",
        ),
        RuleStep(
            id="database_area",
            question="Is the main issue models, queries, transactions, or session management?",
            branches=(
                AnswerBranch(("model", "models", "orm"), "For models, keep persistence mappings consistent with database constraints and avoid putting unrelated service logic into ORM entities."),
                AnswerBranch(("query", "queries", "select", "filter"), "For queries, fetch only the data the operation needs and make ordering and relationship loading explicit where correctness depends on them."),
                AnswerBranch(("transaction", "transactions", "commit", "rollback"), "For transactions, define one application-level unit of work around changes that must succeed or fail together."),
                AnswerBranch(("session", "sessions"), "For session management, create and close sessions at a predictable boundary and avoid passing a live session farther than necessary."),
            ),
            default_response="Keep persistence details behind a narrow repository or unit-of-work boundary so transaction behaviour stays predictable.",
        ),
    ),
)


REDIS_RULE = ConversationRule(
    id="redis",
    keywords=("redis", "cache", "caching"),
    steps=(
        RuleStep(
            id="redis_use",
            question="Are you using Redis for caching, sessions, or coordination?",
            branches=(
                AnswerBranch(("cache", "caching"), "For caching, treat the primary datastore as the source of truth, use bounded TTLs, and make cache failure non-fatal where correctness does not depend on Redis."),
                AnswerBranch(("session", "sessions"), "For sessions, use predictable key namespaces and TTLs, and make expiry behaviour part of the session contract."),
                AnswerBranch(("coordination", "lock", "locks", "queue"), "For coordination, design ownership, expiry, and failure recovery explicitly rather than treating a cache key as a durable transaction."),
            ),
            default_response="Give Redis one clearly defined role so application correctness does not accidentally depend on cache state.",
        ),
    ),
)


DEFAULT_RULES: tuple[ConversationRule, ...] = (
    DOCKER_RULE,
    PYTEST_RULE,
    FASTAPI_RULE,
    DATABASE_RULE,
    REDIS_RULE,
)
