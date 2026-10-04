import Link from "next/link"
import { ArrowRight, Footprints, MapPin } from "lucide-react"
import { highlights, mapReserves, reserves, totals, formatDataThrough } from "@/lib/site-data"
import { groupLabel } from "@/lib/taxa"
import { Legend } from "@/components/site/legend"
import { MapFrame } from "@/components/site/map-frame"

export default function Home() {
  const t = totals()
  const mosaic = reserves.flatMap((r) => highlights(r, 2)).slice(0, 6)

  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden bg-forest text-parchment">
        <div className="topo text-parchment opacity-[0.09]" />
        <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1.1fr_1fr] lg:py-24">
          <div>
            <p className="inline-flex items-center gap-2 rounded-full border border-parchment/25 bg-parchment/5 px-3 py-1 text-xs uppercase tracking-[0.2em] text-parchment/75">
              <Footprints className="h-3.5 w-3.5" /> A field guide to four reserves
            </p>
            <h1 className="mt-6 text-balance font-display text-5xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
              Explore the wild corners of San Diego
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-parchment/80">
              From sea-cliff scrub to tidal salt marsh, UC San Diego looks after four reserves of
              the UC Natural Reserve System. Pick one to see what lives there: every bird, lizard,
              butterfly and wildflower people have recorded.
            </p>
            <dl className="mt-10 grid max-w-md grid-cols-3 gap-6">
              {[
                ["4", "reserves"],
                [t.species.toLocaleString(), "species"],
                [t.observations.toLocaleString(), "records"],
              ].map(([v, k]) => (
                <div key={k}>
                  <dt className="sr-only">{k}</dt>
                  <dd className="font-display text-3xl font-semibold">{v}</dd>
                  <p className="text-xs uppercase tracking-[0.16em] text-parchment/60">{k}</p>
                </div>
              ))}
            </dl>
            <Link
              href="#reserves"
              className="mt-10 inline-flex items-center gap-2 rounded-full bg-parchment px-6 py-3 font-medium text-forest-deep shadow-lg transition hover:gap-3"
            >
              Choose a reserve <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {mosaic.map((s, i) => (
              <figure
                key={s.id}
                className={`group relative overflow-hidden rounded-2xl border border-parchment/15 shadow-xl ${
                  i === 0 ? "col-span-2 row-span-2 aspect-square" : "aspect-square"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={s.photo!.url} alt={s.common ?? s.name} className="h-full w-full object-cover transition duration-700 group-hover:scale-105" />
                <figcaption className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-3 pt-8 text-xs">
                  <span className="block font-medium text-white">{s.common ?? s.name}</span>
                  <span className="block truncate text-[10px] text-white/70">{s.photo!.attribution}</span>
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      </section>

      {/* Map */}
      <section className="paper border-b">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">The map</p>
              <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">From the coast to the mesas</h2>
              <p className="mt-2 max-w-xl text-muted-foreground">
                Four reserves spread across 40 km of coast, canyon and chaparral. Click one to
                step inside.
              </p>
            </div>
            <Legend compact />
          </div>
          <div className="mt-8 overflow-hidden rounded-3xl border shadow-lg">
            <MapFrame reserves={mapReserves()} className="h-[460px] w-full" />
          </div>
        </div>
      </section>

      {/* Reserve cards */}
      <section id="reserves" className="mx-auto max-w-7xl scroll-mt-20 px-4 py-16 sm:px-6">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">The reserves</p>
        <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">Choose where to explore</h2>
        <div className="mt-10 grid gap-6 md:grid-cols-2">
          {reserves.map((r) => {
            const pics = highlights(r, 3)
            const topGroups = Object.entries(r.groups).sort((a, b) => b[1] - a[1]).slice(0, 3)
            return (
              <Link
                key={r.slug}
                href={`/reserves/${r.slug}`}
                className="group relative flex flex-col overflow-hidden rounded-3xl border bg-card shadow-sm transition hover:-translate-y-1 hover:shadow-xl"
              >
                <div className="relative h-44 overflow-hidden" style={{ background: r.color }}>
                  <div className="topo text-white opacity-25" />
                  <div className="absolute inset-0 flex">
                    {pics.map((s) => (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img key={s.id} src={s.photo!.url} alt="" className="h-full flex-1 object-cover opacity-90 transition duration-700 group-hover:scale-105" />
                    ))}
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/10 to-transparent" />
                  <div className="absolute bottom-4 left-5 right-5 text-white">
                    <p className="flex items-center gap-1.5 text-xs uppercase tracking-[0.16em] text-white/80">
                      <MapPin className="h-3.5 w-3.5" /> {r.setting}
                    </p>
                    <h3 className="mt-1 font-display text-2xl font-semibold leading-tight">{r.name}</h3>
                  </div>
                </div>
                <div className="flex flex-1 flex-col gap-4 p-5">
                  <p className="leading-relaxed text-muted-foreground">{r.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {r.habitats.map((h) => (
                      <span key={h} className="rounded-full border px-2.5 py-0.5 text-xs" style={{ borderColor: r.color, color: r.color }}>
                        {h}
                      </span>
                    ))}
                  </div>
                  <div className="mt-auto flex items-end justify-between border-t pt-4">
                    <div className="text-sm">
                      <span className="font-display text-2xl font-semibold">{r.stats.species}</span>{" "}
                      <span className="text-muted-foreground">species recorded</span>
                      <p className="text-xs text-muted-foreground">
                        Most: {topGroups.map(([g, n]) => `${groupLabel(g).toLowerCase()} (${n})`).join(", ")}
                      </p>
                    </div>
                    <span className="inline-flex items-center gap-1 text-sm font-medium text-primary transition group-hover:gap-2">
                      Explore <ArrowRight className="h-4 w-4" />
                    </span>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Legend explainer */}
      <section className="mx-auto max-w-7xl px-4 pb-20 sm:px-6">
        <div className="relative overflow-hidden rounded-3xl bg-secondary p-8 sm:p-10">
          <div className="topo text-primary opacity-[0.07]" />
          <div className="relative grid gap-8 md:grid-cols-[1fr_2fr] md:items-center">
            <h2 className="font-display text-3xl font-semibold">Two ways to know what lives here</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-2xl border border-observed/30 bg-card p-5">
                <p className="font-semibold text-observed">Seen on iNaturalist</p>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  Species visitors and researchers have photographed and logged inside the reserve,
                  current to {formatDataThrough()}. Solid sage markings.
                </p>
              </div>
              <div className="rounded-2xl border-2 border-dashed border-predicted/60 bg-card p-5">
                <p className="font-semibold text-predicted">Model prediction</p>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  Species our model expects but nobody has recorded yet. Dashed ochre markings.
                  Coming once they are checked in the field.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
