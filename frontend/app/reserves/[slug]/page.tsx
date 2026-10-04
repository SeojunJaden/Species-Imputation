import Link from "next/link"
import { notFound } from "next/navigation"
import type { Metadata } from "next"
import { ArrowLeft, ArrowRight, MapPin } from "lucide-react"
import {
  formatDataThrough, getReserve, highlights, mapReserves, orderedGroups, reserves, site,
} from "@/lib/site-data"
import { groupLabel } from "@/lib/taxa"
import { GroupIcon } from "@/components/site/group-icon"
import { Legend } from "@/components/site/legend"
import { MapFrame } from "@/components/site/map-frame"
import { PredictionsPanel } from "@/components/site/predictions-panel"
import { SpeciesExplorer } from "@/components/site/species-explorer"

export function generateStaticParams() {
  return reserves.map((r) => ({ slug: r.slug }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const r = getReserve((await params).slug)
  return { title: r ? `${r.name} · San Diego Reserve Explorer` : "Reserve not found" }
}

export default async function ReservePage({ params }: { params: Promise<{ slug: string }> }) {
  const r = getReserve((await params).slug)
  if (!r) notFound()

  const i = reserves.indexOf(r)
  const prev = reserves[(i + reserves.length - 1) % reserves.length]
  const next = reserves[(i + 1) % reserves.length]
  const groups = orderedGroups(r)
  const stars = highlights(r, 4)

  return (
    <>
      {/* Reserve hero */}
      <section className="relative overflow-hidden text-parchment" style={{ background: `linear-gradient(135deg, hsl(var(--forest-deep)), ${r.color})` }}>
        <div className="topo text-parchment opacity-[0.1]" />
        <div className="relative mx-auto max-w-7xl px-4 pb-12 pt-8 sm:px-6">
          <div className="flex items-center justify-between text-sm">
            <Link href="/#reserves" className="inline-flex items-center gap-1.5 text-parchment/75 hover:text-parchment">
              <ArrowLeft className="h-4 w-4" /> All reserves
            </Link>
            <div className="flex gap-2">
              <Link href={`/reserves/${prev.slug}`} className="rounded-full border border-parchment/25 px-3 py-1 text-parchment/80 hover:bg-parchment/10">
                ← {prev.shortName}
              </Link>
              <Link href={`/reserves/${next.slug}`} className="rounded-full border border-parchment/25 px-3 py-1 text-parchment/80 hover:bg-parchment/10">
                {next.shortName} →
              </Link>
            </div>
          </div>

          <div className="mt-10 grid gap-10 lg:grid-cols-[1.15fr_1fr] lg:items-center">
            <div>
              <p className="flex items-center gap-1.5 text-xs uppercase tracking-[0.2em] text-parchment/70">
                <MapPin className="h-3.5 w-3.5" /> {r.setting} · about {r.areaHa} ha
              </p>
              <h1 className="mt-3 text-balance font-display text-4xl font-semibold leading-tight sm:text-5xl">{r.name}</h1>
              <p className="mt-4 max-w-xl text-lg leading-relaxed text-parchment/85">{r.description}</p>
              <div className="mt-5 flex flex-wrap gap-2">
                {r.habitats.map((h) => (
                  <span key={h} className="rounded-full border border-parchment/30 bg-parchment/10 px-3 py-1 text-xs">
                    {h}
                  </span>
                ))}
              </div>
              <dl className="mt-8 grid grid-cols-3 gap-4 sm:max-w-md">
                {[
                  [r.stats.species.toLocaleString(), "species"],
                  [r.stats.observations.toLocaleString(), "records"],
                  [r.stats.observers.toLocaleString(), "observers"],
                ].map(([v, k]) => (
                  <div key={k} className="rounded-2xl border border-parchment/15 bg-black/10 p-3">
                    <dd className="font-display text-2xl font-semibold">{v}</dd>
                    <dt className="text-[11px] uppercase tracking-[0.14em] text-parchment/65">{k}</dt>
                  </div>
                ))}
              </dl>
            </div>
            <div className="overflow-hidden rounded-3xl border border-parchment/20 shadow-2xl">
              <MapFrame reserves={mapReserves()} focus={r.slug} className="h-[340px] w-full" />
            </div>
          </div>
        </div>
      </section>

      {/* Who lives here at a glance */}
      <section className="paper border-b">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
          <h2 className="font-display text-2xl font-semibold">Who lives here</h2>
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {groups.map(([g, n]) => (
              <div key={g} className="flex items-center gap-3 rounded-2xl border bg-card px-4 py-3 shadow-sm">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-observed-soft text-observed">
                  <GroupIcon group={g} className="h-4.5 w-4.5" />
                </span>
                <span className="leading-tight">
                  <span className="block font-display text-xl font-semibold">{n}</span>
                  <span className="text-xs text-muted-foreground">{groupLabel(g)}</span>
                </span>
              </div>
            ))}
          </div>
          {stars.length > 0 && (
            <p className="mt-5 text-sm text-muted-foreground">
              Often seen here:{" "}
              {stars.map((s, k) => (
                <span key={s.id}>
                  <span className="font-medium text-foreground">{s.common ?? s.name}</span>
                  {k < stars.length - 1 ? ", " : "."}
                </span>
              ))}
            </p>
          )}
        </div>
      </section>

      {/* Species explorer */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-observed">Seen on iNaturalist</p>
            <h2 className="mt-2 font-display text-3xl font-semibold sm:text-4xl">What you might see</h2>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              Every species recorded inside the reserve, current to {formatDataThrough()}. Animals
              come first; switch groups, search, or pick a month to see what turns up in that season.
            </p>
          </div>
          <Legend />
        </div>
        <div className="mt-8">
          <SpeciesExplorer species={r.species} groups={groups} />
        </div>
      </section>

      {/* Model predictions */}
      <section className="mx-auto max-w-7xl px-4 pb-20 sm:px-6">
        <PredictionsPanel status={site.predictionsStatus} predicted={r.predicted} reserveName={r.shortName} />
        <div className="mt-10 flex justify-between text-sm">
          <Link href={`/reserves/${prev.slug}`} className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline">
            <ArrowLeft className="h-4 w-4" /> {prev.name}
          </Link>
          <Link href={`/reserves/${next.slug}`} className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline">
            {next.name} <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </>
  )
}
