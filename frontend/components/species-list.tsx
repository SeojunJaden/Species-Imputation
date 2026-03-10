"use client"

import React from "react"

import { useState, useMemo } from "react"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Species, PredictedSpecies } from "@/lib/reserve-data"
import { taxonColors } from "@/lib/reserve-data"
import {
  Search,
  Leaf,
  Bird,
  Squirrel,
  Bug,
  Droplets,
  Shell,
  TreeDeciduous,
  Fish,
  CheckCircle2,
  AlertCircle,
  Eye,
} from "lucide-react"

const taxonIconMap: Record<string, React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  Plantae: Leaf,
  Aves: Bird,
  Mammalia: Squirrel,
  Reptilia: Bug,
  Amphibia: Droplets,
  Insecta: Bug,
  Arachnida: Bug,
  Mollusca: Shell,
  Fungi: TreeDeciduous,
  Actinopterygii: Fish,
}

interface ObservedSpeciesListProps {
  species: Species[]
  reserveName: string
}

export function ObservedSpeciesList({ species, reserveName }: ObservedSpeciesListProps) {
  const [search, setSearch] = useState("")
  const [taxonFilter, setTaxonFilter] = useState<string | null>(null)

  const taxonCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    species.forEach((s) => {
      counts[s.iconicTaxon] = (counts[s.iconicTaxon] || 0) + 1
    })
    return counts
  }, [species])

  const filtered = useMemo(() => {
    return species
      .filter((s) => {
        const matchesSearch =
          s.commonName.toLowerCase().includes(search.toLowerCase()) ||
          s.scientificName.toLowerCase().includes(search.toLowerCase())
        const matchesTaxon = !taxonFilter || s.iconicTaxon === taxonFilter
        return matchesSearch && matchesTaxon
      })
      .sort((a, b) => b.observationCount - a.observationCount)
  }, [species, search, taxonFilter])

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/15">
          <CheckCircle2 className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">Observed Species</h3>
          <p className="text-xs text-muted-foreground">
            {species.length} species in {reserveName}
          </p>
        </div>
      </div>

      <div className="border-b border-border px-4 py-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search species..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 border-border bg-secondary pl-8 text-xs text-foreground placeholder:text-muted-foreground focus-visible:ring-primary"
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-1 border-b border-border px-4 py-2">
        <button
          type="button"
          onClick={() => setTaxonFilter(null)}
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors ${
            !taxonFilter
              ? "bg-primary/20 text-primary"
              : "bg-secondary text-muted-foreground hover:text-foreground"
          }`}
        >
          All
        </button>
        {Object.entries(taxonCounts)
          .sort((a, b) => b[1] - a[1])
          .map(([taxon, count]) => (
            <button
              key={taxon}
              type="button"
              onClick={() => setTaxonFilter(taxon === taxonFilter ? null : taxon)}
              className={`flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors ${
                taxonFilter === taxon
                  ? "bg-primary/20 text-primary"
                  : "bg-secondary text-muted-foreground hover:text-foreground"
              }`}
            >
              {taxon}
              <span className="opacity-60">{count}</span>
            </button>
          ))}
      </div>

      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-1 p-2">
          {filtered.map((s) => {
            const IconComponent = taxonIconMap[s.iconicTaxon] || Bug
            return (
              <div
                key={s.id}
                className="group flex items-start gap-3 rounded-lg px-3 py-2.5 transition-colors hover:bg-secondary"
              >
                <div
                  className="mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md"
                  style={{ backgroundColor: `${taxonColors[s.iconicTaxon]}15` }}
                >
                  <IconComponent
                    className="h-3.5 w-3.5"
                    style={{ color: taxonColors[s.iconicTaxon] }}
                  />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium leading-tight text-foreground">
                    {s.commonName}
                  </p>
                  <p className="text-xs italic text-muted-foreground">{s.scientificName}</p>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Eye className="h-3 w-3" />
                      {s.observationCount}
                    </span>
                    <Badge
                      variant="outline"
                      className={`h-4 border-0 px-1.5 text-[10px] font-medium ${
                        s.qualityGrade === "research"
                          ? "bg-primary/10 text-primary"
                          : s.qualityGrade === "needs_id"
                            ? "bg-accent/10 text-accent"
                            : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {s.qualityGrade === "research"
                        ? "Research"
                        : s.qualityGrade === "needs_id"
                          ? "Needs ID"
                          : "Casual"}
                    </Badge>
                  </div>
                </div>
              </div>
            )
          })}
          {filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Search className="mb-2 h-8 w-8 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No species found</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

interface PredictedSpeciesListProps {
  species: PredictedSpecies[]
  reserveName: string
}

export function PredictedSpeciesList({ species, reserveName }: PredictedSpeciesListProps) {
  const [search, setSearch] = useState("")

  const filtered = useMemo(() => {
    return species
      .filter(
        (s) =>
          s.commonName.toLowerCase().includes(search.toLowerCase()) ||
          s.scientificName.toLowerCase().includes(search.toLowerCase())
      )
      .sort((a, b) => b.probability - a.probability)
  }, [species, search])

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent/15">
          <AlertCircle className="h-4 w-4 text-accent" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">Predicted Species</h3>
          <p className="text-xs text-muted-foreground">
            {species.length} predictions for {reserveName}
          </p>
        </div>
      </div>

      <div className="border-b border-border px-4 py-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search predictions..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 border-border bg-secondary pl-8 text-xs text-foreground placeholder:text-muted-foreground focus-visible:ring-primary"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-1 p-2">
          {filtered.map((s) => {
            const IconComponent = taxonIconMap[s.iconicTaxon] || Bug
            const probabilityPercent = Math.round(s.probability * 100)
            const probabilityColor =
              probabilityPercent >= 75
                ? "text-primary"
                : probabilityPercent >= 50
                  ? "text-accent"
                  : "text-destructive"
            const barColor =
              probabilityPercent >= 75
                ? "bg-primary"
                : probabilityPercent >= 50
                  ? "bg-accent"
                  : "bg-destructive"
            return (
              <div
                key={s.id}
                className="group rounded-lg px-3 py-2.5 transition-colors hover:bg-secondary"
              >
                <div className="flex items-start gap-3">
                  <div
                    className="mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md"
                    style={{ backgroundColor: `${taxonColors[s.iconicTaxon]}15` }}
                  >
                    <IconComponent
                      className="h-3.5 w-3.5"
                      style={{ color: taxonColors[s.iconicTaxon] }}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium leading-tight text-foreground">
                        {s.commonName}
                      </p>
                      <span className={`flex-shrink-0 text-sm font-bold tabular-nums ${probabilityColor}`}>
                        {probabilityPercent}%
                      </span>
                    </div>
                    <p className="text-xs italic text-muted-foreground">{s.scientificName}</p>

                    <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-secondary">
                      <div
                        className={`h-full rounded-full ${barColor} transition-all`}
                        style={{ width: `${probabilityPercent}%` }}
                      />
                    </div>

                    <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                      {s.reason}
                    </p>
                    <p className="mt-0.5 text-[10px] text-muted-foreground/70">
                      Nearest: {s.nearestObservedReserve} ({s.distanceKm} km)
                    </p>
                  </div>
                </div>
              </div>
            )
          })}
          {filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Search className="mb-2 h-8 w-8 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No predictions found</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}