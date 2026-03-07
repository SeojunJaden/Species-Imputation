"use client"

import { useState, useCallback } from "react"
import dynamic from "next/dynamic"
import { reserves } from "@/lib/reserve-data"
import { ReserveSelector, StatsBar } from "@/components/reserve-selector"
import { ObservedSpeciesList, PredictedSpeciesList } from "@/components/species-list"
import { MapPin, Layers } from "lucide-react"

const ReserveMap = dynamic(
  () => import("@/components/reserve-map").then((m) => ({ default: m.ReserveMap })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center rounded-lg bg-card">
        <div className="flex flex-col items-center gap-2">
          <MapPin className="h-8 w-8 animate-pulse text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading map...</p>
        </div>
      </div>
    ),
  }
)

export default function DashboardPage() {
  const [selectedReserveId, setSelectedReserveId] = useState<string>("scripps")

  const selectedReserve = reserves.find((r) => r.id === selectedReserveId) || reserves[0]

  const handleSelectReserve = useCallback((id: string) => {
    setSelectedReserveId(id)
  }, [])

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      {/* Header */}
      <header className="flex flex-shrink-0 items-center justify-between border-b border-border bg-card px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15">
            <Layers className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-foreground">
              UCSD Reserve Species Explorer
            </h1>
            <p className="text-xs text-muted-foreground">
              UC San Diego Natural Reserve System &middot; Biodiversity Dashboard
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-4 md:flex">
          <StatsBar reserve={selectedReserve} />
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar - Reserve Selector */}
        <aside className="flex w-64 flex-shrink-0 flex-col border-r border-border bg-card lg:w-72">
          <div className="border-b border-border px-4 py-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Reserves
            </h2>
          </div>
          <div className="flex-1 overflow-auto p-3">
            <ReserveSelector
              reserves={reserves}
              selectedReserve={selectedReserveId}
              onSelectReserve={handleSelectReserve}
            />
          </div>
          <div className="border-t border-border px-4 py-3">
            <div className="rounded-lg bg-secondary p-3">
              <h3 className="text-xs font-semibold text-foreground">{selectedReserve.name}</h3>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {selectedReserve.description}
              </p>
            </div>
          </div>
        </aside>

        {/* Center - Map */}
        <main className="flex flex-1 flex-col">
          {/* Stats bar for mobile */}
          <div className="flex items-center overflow-x-auto border-b border-border bg-card px-4 py-2 md:hidden">
            <StatsBar reserve={selectedReserve} />
          </div>
          <div className="flex-1 p-2">
            <ReserveMap
              reserves={reserves}
              selectedReserve={selectedReserveId}
              onSelectReserve={handleSelectReserve}
            />
          </div>
        </main>

        {/* Right Panel - Species Lists */}
        <aside className="flex w-80 flex-shrink-0 flex-col border-l border-border bg-card lg:w-96">
          <div className="flex h-1/2 flex-col border-b border-border">
            <ObservedSpeciesList
              species={selectedReserve.observedSpecies}
              reserveName={selectedReserve.shortName}
            />
          </div>
          <div className="flex h-1/2 flex-col">
            <PredictedSpeciesList
              species={selectedReserve.predictedSpecies}
              reserveName={selectedReserve.shortName}
            />
          </div>
        </aside>
      </div>
    </div>
  )
}