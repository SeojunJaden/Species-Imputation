"use client"

import { useEffect, useRef } from "react"
import type { Reserve } from "@/lib/reserve-data"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

interface ReserveMapProps {
  reserves: Reserve[]
  selectedReserve: string | null
  onSelectReserve: (id: string) => void
}

export function ReserveMap({ reserves, selectedReserve, onSelectReserve }: ReserveMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const polygonsRef = useRef<L.Polygon[]>([])
  const labelsRef = useRef<L.Marker[]>([])

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return

    const map = L.map(mapRef.current, {
      center: [32.82, -117.18],
      zoom: 12,
      zoomControl: false,
      attributionControl: false,
    })

    L.control.zoom({ position: "bottomright" }).addTo(map)

    L.control.attribution({ position: "bottomleft", prefix: false }).addTo(map)

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map)

    mapInstanceRef.current = map

    return () => {
      map.remove()
      mapInstanceRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map) return

    polygonsRef.current.forEach((p) => p.remove())
    polygonsRef.current = []
    labelsRef.current.forEach((l) => l.remove())
    labelsRef.current = []

    reserves.forEach((reserve) => {
      const isSelected = selectedReserve === reserve.id

      const polygon = L.polygon(reserve.boundary, {
        color: isSelected ? "#ffffff" : reserve.color,
        weight: isSelected ? 3 : 2,
        opacity: isSelected ? 1 : 0.8,
        fillColor: reserve.color,
        fillOpacity: isSelected ? 0.35 : 0.15,
        dashArray: isSelected ? undefined : "6 4",
      })

      polygon.bindTooltip(
        `<div style="font-family: system-ui; font-size: 13px; font-weight: 600; color: #e2e8f0;">${reserve.shortName}</div>
         <div style="font-family: system-ui; font-size: 11px; color: #94a3b8;">${reserve.totalSpecies} species &middot; ${reserve.totalObservations.toLocaleString()} obs</div>
         <div style="font-family: system-ui; font-size: 10px; color: #64748b; margin-top: 2px;">${reserve.areaHa} ha &middot; ${reserve.ecosystem}</div>`,
        {
          permanent: false,
          direction: "top",
          className: "reserve-tooltip",
          sticky: true,
        },
      )

      polygon.on("click", () => {
        onSelectReserve(reserve.id)
      })

      polygon.on("mouseover", () => {
        if (!isSelected) {
          polygon.setStyle({
            fillOpacity: 0.3,
            weight: 3,
            opacity: 1,
          })
        }
      })

      polygon.on("mouseout", () => {
        if (!isSelected) {
          polygon.setStyle({
            fillOpacity: 0.15,
            weight: 2,
            opacity: 0.8,
          })
        }
      })

      polygon.addTo(map)
      polygonsRef.current.push(polygon)

      const label = L.marker([reserve.latitude, reserve.longitude], {
        icon: L.divIcon({
          className: "reserve-label",
          html: `<span style="
            font-family: system-ui;
            font-size: 11px;
            font-weight: 600;
            color: ${reserve.color};
            text-shadow: 0 0 6px rgba(0,0,0,0.9), 0 0 3px rgba(0,0,0,0.9);
            white-space: nowrap;
            pointer-events: none;
            letter-spacing: 0.5px;
          ">${reserve.shortName}</span>`,
          iconSize: [0, 0],
          iconAnchor: [0, 0],
        }),
        interactive: false,
      })
      label.addTo(map)
      labelsRef.current.push(label)
    })
  }, [reserves, selectedReserve, onSelectReserve])

  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !selectedReserve) return

    const reserve = reserves.find((r) => r.id === selectedReserve)
    if (reserve) {
      const bounds = L.latLngBounds(reserve.boundary)
      map.flyToBounds(bounds, { padding: [50, 50], duration: 0.8 })
    }
  }, [selectedReserve, reserves])

  return (
    <div className="relative h-full w-full">
      <div ref={mapRef} className="h-full w-full rounded-lg" />
      <style jsx global>{`
        .reserve-tooltip {
          background: hsl(200, 18%, 10%) !important;
          border: 1px solid hsl(200, 12%, 22%) !important;
          border-radius: 8px !important;
          padding: 8px 12px !important;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        }
        .reserve-tooltip::before {
          border-top-color: hsl(200, 12%, 22%) !important;
        }
        .leaflet-control-zoom a {
          background: hsl(200, 18%, 10%) !important;
          color: hsl(180, 10%, 85%) !important;
          border-color: hsl(200, 12%, 18%) !important;
        }
        .leaflet-control-zoom a:hover {
          background: hsl(200, 14%, 16%) !important;
        }
        .reserve-label {
          background: none !important;
          border: none !important;
          box-shadow: none !important;
        }
      `}</style>
    </div>
  )
}