"""Test prompt caching across model families through LiteLLM → OpenRouter.

Tests Anthropic (Claude), Google (Gemini), and verifies cache_control
breakpoints propagate correctly. Queries Langfuse for server-side
cost/cache metrics after each test run.

Usage:
    uv run python scripts/test_prompt_caching.py
    uv run python scripts/test_prompt_caching.py --model gemini
    uv run python scripts/test_prompt_caching.py --model sonnet --langfuse
    uv run python scripts/test_prompt_caching.py --all --langfuse
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import click
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Large system prompt (~6000 tokens) — exceeds all provider minimums:
# - Anthropic Sonnet/Opus: 1024 min
# - Anthropic Haiku 4.5/Opus 4.5: 4096 min
# - Gemini: 4096 min (typically)
# ---------------------------------------------------------------------------
_IMAS_REFERENCE = """
## IMAS IDS Structure Reference

The IMAS (Integrated Modelling & Analysis Suite) defines Interface Data
Structures (IDS) for standardized fusion data exchange across facilities.

### equilibrium
Plasma equilibrium reconstruction (EFIT, LIUQE, etc.):
- time_slice[].profiles_1d.psi — Poloidal flux
- time_slice[].profiles_1d.pressure — Plasma pressure
- time_slice[].profiles_2d[].psi — 2D poloidal flux map
- time_slice[].boundary.outline.r — Plasma boundary R coordinates
- time_slice[].boundary.outline.z — Plasma boundary Z coordinates
- time_slice[].global_quantities.ip — Plasma current
- time_slice[].global_quantities.magnetic_axis.r — Magnetic axis R
- time_slice[].global_quantities.magnetic_axis.z — Magnetic axis Z
- time_slice[].global_quantities.li_3 — Internal inductance
- time_slice[].global_quantities.beta_pol — Poloidal beta
- time_slice[].global_quantities.beta_tor — Toroidal beta
- time_slice[].global_quantities.energy_mhd — MHD stored energy
- time_slice[].profiles_1d.q — Safety factor profile
- time_slice[].profiles_1d.j_tor — Toroidal current density
- time_slice[].profiles_1d.rho_tor_norm — Normalized toroidal flux coord

### core_profiles
Core plasma profiles:
- profiles_1d[].electrons.temperature — Electron temperature Te
- profiles_1d[].electrons.density — Electron density ne
- profiles_1d[].electrons.density_thermal — Thermal electron density
- profiles_1d[].ion[].temperature — Ion temperature Ti
- profiles_1d[].ion[].density — Ion density
- profiles_1d[].ion[].density_thermal — Thermal ion density
- profiles_1d[].ion[].element[].z_n — Nuclear charge
- profiles_1d[].ion[].element[].a — Atomic mass
- profiles_1d[].grid.rho_tor_norm — Normalized toroidal flux coordinate
- profiles_1d[].zeff — Effective charge
- profiles_1d[].rotation.toroidal — Toroidal rotation velocity

### magnetics
Magnetic diagnostics:
- flux_loop[].flux.data — Flux loop measurements
- bpol_probe[].field.data — Poloidal field probe data
- method[].ip.data — Plasma current from magnetics
- ip.data — Total plasma current
- diamagnetic_flux.data — Diamagnetic flux measurement

### pf_active
Active poloidal field coils:
- coil[].current.data — Coil currents
- coil[].element[].geometry.rectangle — Coil geometry

### wall
First wall and vessel structures:
- description_2d[].limiter.unit[].outline.r — Wall R contour
- description_2d[].limiter.unit[].outline.z — Wall Z contour
- description_2d[].vessel.unit[].outline — Vessel outline

### nbi
Neutral Beam Injection:
- unit[].power_launched.data — NBI power
- unit[].energy.data — Beam energy
- unit[].species.a — Ion mass
- unit[].species.z_n — Ion charge

### ec_launchers
Electron Cyclotron launchers:
- beam[].power_launched.data — EC power
- beam[].steering_angle_pol — Poloidal steering angle
- beam[].steering_angle_tor — Toroidal steering angle
- beam[].frequency.data — Gyrotron frequency

### thomson_scattering
Thomson scattering diagnostic:
- channel[].n_e.data — Electron density
- channel[].t_e.data — Electron temperature
- channel[].position.r — Channel R position
- channel[].position.z — Channel Z position
- channel[].position.phi — Channel phi position

### interferometer
Interferometry diagnostic:
- channel[].n_e_line.data — Line-integrated density
- channel[].n_e_line_average.data — Line-average density

### charge_exchange
Charge exchange spectroscopy:
- channel[].ion[].temperature — Ion temperature from CX
- channel[].ion[].velocity.toroidal — Toroidal rotation
- channel[].ion[].velocity.poloidal — Poloidal rotation

### bolometer
Bolometry system:
- channel[].power.data — Radiated power per channel
- channel[].etendue — Etendue of the channel

### ece
Electron cyclotron emission:
- channel[].t_e.data — Electron temperature
- channel[].frequency.data — ECE frequency
- channel[].position.r — Measurement position R

### soft_x_rays
Soft X-ray diagnostic:
- channel[].brightness.data — Measured brightness
- channel[].filter[].material — Filter material

### summary
Global plasma summary:
- global_quantities.ip.value — Plasma current
- global_quantities.energy_mhd.value — Stored energy
- global_quantities.beta_pol.value — Poloidal beta
- global_quantities.li.value — Internal inductance
- global_quantities.v_loop.value — Loop voltage

## MDSplus Tree Patterns
Common MDSplus tree names: tcv_shot, atlas, magnetics, results, ecrh, trans, cxrs
TDI expressions: \\tree::node:path, _sig=data(\\tree::path)
MDSplus Python: import MDSplus; t = MDSplus.Tree('tcv_shot', shot)

## COCOS (COordinate COnventions Standard)
- COCOS 1-8: Different sign conventions for B_phi, I_p, q, psi
- COCOS 11: ITER standard (positive Ip, positive q)
- cocos_transform(ids, from_cocos, to_cocos) for conversion
- Key quantities: B_phi sign, I_p sign, q sign, psi boundary conditions

## Scoring Guidelines
- 0.9-1.0: Direct IMAS IDS manipulation, IDS read/write, imas Python bindings
- 0.7-0.8: MDSplus data access, equilibrium reconstruction codes
- 0.5-0.6: General analysis codes that process fusion data
- 0.3-0.4: Visualization tools, plotting utilities for fusion data
- 0.1-0.2: Generic scientific computing with potential fusion use
- 0.0: No connection to fusion or plasma physics
"""

SYSTEM_PROMPT = (
    "You are an expert in IMAS (Integrated Modelling & Analysis Suite) "
    "for fusion research. You understand data dictionaries, IDS structures, "
    "COCOS conventions, MDSplus trees, and facility-specific data systems "
    "across TCV, JET, ITER, and JT-60SA.\n\n"
    + _IMAS_REFERENCE
)

# Different user questions so we test cache reuse across varying user content
USER_QUESTIONS = [
    "What is an IDS in IMAS? Answer in one sentence.",
    "Explain COCOS conventions in one sentence.",
    "What is Thomson scattering used for? One sentence.",
    "Name three IMAS diagnostics. One sentence.",
]

# Model presets
MODEL_PRESETS: dict[str, str] = {
    "sonnet": "anthropic/claude-sonnet-4.5",
    "haiku": "anthropic/claude-haiku-4.5",
    "opus": "anthropic/claude-opus-4.6",
    "gemini": "google/gemini-3-flash-preview",
}


@dataclass
class CallResult:
    """Result from a single LLM call."""

    model: str
    call_num: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    cost: float
    answer: str
    generation_id: str = ""
    elapsed: float = 0.0


@dataclass
class TestReport:
    """Summary report for a model's cache test."""

    model: str
    calls: list[CallResult] = field(default_factory=list)

    @property
    def cache_hits(self) -> int:
        return sum(1 for c in self.calls if c.cached_tokens > 0)

    @property
    def cache_writes(self) -> int:
        return sum(1 for c in self.calls if c.cache_write_tokens > 0)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)


def _make_call(
    model: str,
    call_num: int,
    question: str,
    pin_provider: bool = False,
) -> CallResult:
    """Make a single LLM call through our discovery infrastructure."""
    import litellm

    from imas_codex.discovery.base.llm import (
        ensure_openrouter_prefix,
        extract_cost,
        get_api_key,
        inject_cache_control,
        suppress_litellm_noise,
    )

    suppress_litellm_noise()
    model_id = ensure_openrouter_prefix(model)
    api_key = get_api_key()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    messages = inject_cache_control(messages)

    kwargs: dict = {
        "model": model_id,
        "api_key": api_key,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.3,
        "timeout": 60,
    }
    if pin_provider:
        provider = "Anthropic" if "claude" in model.lower() else "Google"
        kwargs["extra_body"] = {"provider": {"order": [provider]}}

    t0 = time.monotonic()
    response = litellm.completion(**kwargs)
    elapsed = time.monotonic() - t0

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)

    # Extract generation ID from response headers if available
    gen_id = ""
    if hasattr(response, "_hidden_params"):
        hp = response._hidden_params
        gen_id = hp.get("additional_headers", {}).get("x-generation-id", "")
        if not gen_id:
            gen_id = str(hp.get("response_cost", ""))

    return CallResult(
        model=model,
        call_num=call_num,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        cache_write_tokens=getattr(details, "cache_write_tokens", 0) or 0,
        cost=extract_cost(response),
        answer=response.choices[0].message.content.strip()[:80],
        generation_id=gen_id,
        elapsed=elapsed,
    )


def _run_model_test(
    model: str,
    num_calls: int = 4,
    pause: float = 3.0,
    pin_provider: bool = False,
) -> TestReport:
    """Run cache test for a single model."""
    report = TestReport(model=model)

    for i in range(num_calls):
        question = USER_QUESTIONS[i % len(USER_QUESTIONS)]
        if i > 0:
            click.echo(f"  Pausing {pause:.0f}s...")
            time.sleep(pause)

        try:
            result = _make_call(model, i + 1, question, pin_provider)
            report.calls.append(result)
            # Determine cache status symbol
            if result.cached_tokens > 0:
                status = click.style("HIT", fg="green", bold=True)
            elif result.cache_write_tokens > 0:
                status = click.style("WRITE", fg="yellow")
            else:
                status = click.style("MISS", fg="red")

            click.echo(
                f"  Call {i + 1}: {status}  "
                f"prompt={result.prompt_tokens}  "
                f"cached={result.cached_tokens}  "
                f"write={result.cache_write_tokens}  "
                f"cost=${result.cost:.6f}  "
                f"time={result.elapsed:.1f}s  "
                f"| {result.answer[:60]}"
            )
        except Exception as e:
            click.echo(click.style(f"  Call {i + 1}: ERROR — {e}", fg="red"))

    return report


def _query_langfuse(reports: list[TestReport]) -> None:
    """Query Langfuse API for trace details on recent generations."""
    try:
        from langfuse import Langfuse
    except ImportError:
        click.echo("\n  langfuse package not installed — skipping API query")
        click.echo("  Install: uv pip install langfuse")
        return

    secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    base = os.environ.get("LANGFUSE_BASE_URL", "").strip()

    if not (secret and public):
        click.echo("\n  Langfuse keys not configured — skipping API query")
        return

    click.echo("\n" + "=" * 70)
    click.echo("LANGFUSE TRACE ANALYSIS")
    click.echo("=" * 70)

    try:
        lf = Langfuse(
            secret_key=secret,
            public_key=public,
            host=base or "https://cloud.langfuse.com",
        )
        # Flush any pending events first
        lf.flush()
        time.sleep(2)  # Let Langfuse ingest

        # Get recent generations
        generations = lf.get_generations(limit=20)

        if not generations or not generations.data:
            click.echo("  No recent generations found in Langfuse")
            return

        click.echo(f"\n  Found {len(generations.data)} recent generations:\n")
        click.echo(
            f"  {'Model':<40} {'Prompt':>7} {'Compl':>6} "
            f"{'Cached':>7} {'Write':>6} {'Cost':>10} {'Created'}"
        )
        click.echo("  " + "-" * 110)

        for gen in generations.data[:15]:
            model_name = gen.model or "unknown"
            usage = gen.usage or {}

            # Langfuse stores usage differently depending on provider
            prompt_t = getattr(usage, "input", 0) or getattr(
                usage, "prompt_tokens", 0
            )
            compl_t = getattr(usage, "output", 0) or getattr(
                usage, "completion_tokens", 0
            )
            # Check for cache info in usage details
            cache_read = getattr(usage, "input_cached", 0) or 0
            cache_write = getattr(usage, "cache_write", 0) or 0

            cost = gen.calculated_total_cost or 0
            created = str(gen.start_time or "")[:19]

            click.echo(
                f"  {model_name:<40} {prompt_t:>7} {compl_t:>6} "
                f"{cache_read:>7} {cache_write:>6} "
                f"${cost:>9.6f} {created}"
            )

    except Exception as e:
        click.echo(f"  Langfuse query error: {e}")


def _print_summary(reports: list[TestReport]) -> None:
    """Print summary table."""
    click.echo("\n" + "=" * 70)
    click.echo("SUMMARY")
    click.echo("=" * 70)

    click.echo(
        f"\n  {'Model':<35} {'Calls':>5} {'Hits':>5} "
        f"{'Writes':>6} {'Total $':>10} {'Hit Rate':>8}"
    )
    click.echo("  " + "-" * 80)

    for r in reports:
        n = len(r.calls)
        hit_rate = f"{r.cache_hits / n * 100:.0f}%" if n > 0 else "N/A"
        click.echo(
            f"  {r.model:<35} {n:>5} {r.cache_hits:>5} "
            f"{r.cache_writes:>6} ${r.total_cost:>9.6f} {hit_rate:>8}"
        )

    # Print cache economics
    click.echo("\n  Cache Economics (per OpenRouter docs):")
    click.echo("  " + "-" * 55)
    click.echo(
        "  Anthropic: read=0.1× input  write=1.25× (5min) / 2× (1hr)"
    )
    click.echo("  Gemini:    read=0.25× input  write≈input+storage (5min)")
    click.echo(
        "  Gemini implicit caching is automatic (no breakpoints needed)"
    )
    click.echo("  cache_control breakpoints give explicit control over what's cached")


@click.command()
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Model preset (sonnet/haiku/opus/gemini) or full model ID. Repeatable.",
)
@click.option("--all", "run_all", is_flag=True, help="Test all model presets")
@click.option("--calls", "-n", default=4, help="Number of calls per model")
@click.option("--pause", "-p", default=3.0, help="Seconds between calls")
@click.option(
    "--pin-provider",
    is_flag=True,
    help="Pin to native provider (Anthropic/Google)",
)
@click.option("--langfuse", "use_langfuse", is_flag=True, help="Query Langfuse after")
def main(
    model: tuple[str, ...],
    run_all: bool,
    calls: int,
    pause: float,
    pin_provider: bool,
    use_langfuse: bool,
) -> None:
    """Test prompt caching across model families.

    Sends repeated requests with the same system prompt and different user
    questions, checking for cache hits on subsequent calls.
    """
    if run_all:
        models = list(MODEL_PRESETS.values())
    elif model:
        models = [MODEL_PRESETS.get(m, m) for m in model]
    else:
        # Default: test sonnet + gemini (the two workhorses)
        models = [MODEL_PRESETS["sonnet"], MODEL_PRESETS["gemini"]]

    click.echo(f"System prompt: ~{len(SYSTEM_PROMPT.split())} words")
    click.echo(f"Calls per model: {calls}")
    click.echo(f"Pause between calls: {pause}s")
    click.echo(f"Pin provider: {pin_provider}")
    click.echo()

    reports: list[TestReport] = []
    for m in models:
        click.echo(f"{'=' * 70}")
        click.echo(f"Testing: {m}")
        click.echo(f"{'=' * 70}")
        report = _run_model_test(m, num_calls=calls, pause=pause, pin_provider=pin_provider)
        reports.append(report)
        click.echo()

    _print_summary(reports)

    if use_langfuse:
        _query_langfuse(reports)


if __name__ == "__main__":
    main()
