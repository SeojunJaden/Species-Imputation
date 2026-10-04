"use client"

import dynamic from "next/dynamic"
import { Map as MapIcon } from "lucide-react"
import type { MapReserve } from "@/components/site/reserve-map"

// Leaflet touches `window`, so the map only renders in the browser.
const ReserveMap = dynamic(() => import("@/components/site/reserve-map"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-muted">
      <MapIcon className="h-8 w-8 animate-pulse text-muted-foreground" />
    </div>
  ),
})

export function MapFrame(props: { reserves: MapReserve[]; focus?: string; className?: string }) {
  return <ReserveMap {...props} />
}
