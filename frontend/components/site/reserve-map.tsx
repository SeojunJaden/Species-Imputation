"use client"

import { useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

export interface MapReserve {
  slug: string
  shortName: string
  color: string
  boundary: [number, number][]
  center: [number, number]
  speciesCount: number
}

interface Props {
  reserves: MapReserve[]
  /** Zoom to this reserve and draw the others faintly. Omit for the overview. */
  focus?: string
  className?: string
}

const TERRAIN = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
const SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
const ATTRIBUTION = "Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics, and the GIS User Community"

export default function ReserveMap({ reserves, focus, className }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const router = useRouter()

  useEffect(() => {
    if (!el.current) return
    const map = L.map(el.current, {
      zoomControl: false,
      scrollWheelZoom: false,
      attributionControl: true,
    })
    L.control.zoom({ position: "bottomright" }).addTo(map)
    map.attributionControl.setPrefix(false)

    const terrain = L.tileLayer(TERRAIN, { attribution: ATTRIBUTION, maxZoom: 18 })
    const satellite = L.tileLayer(SATELLITE, { attribution: ATTRIBUTION, maxZoom: 18 })
    ;(focus ? satellite : terrain).addTo(map)
    L.control.layers({ Terrain: terrain, Satellite: satellite }, undefined, { position: "topright" }).addTo(map)

    const all = L.latLngBounds([])
    let target: L.LatLngBounds | null = null

    reserves.forEach((r) => {
      const isFocus = r.slug === focus
      const dim = focus && !isFocus
      const poly = L.polygon(r.boundary, {
        color: isFocus ? "#fffaf0" : r.color,
        weight: isFocus ? 3 : 2,
        fillColor: r.color,
        fillOpacity: isFocus ? 0.28 : dim ? 0.12 : 0.35,
        opacity: dim ? 0.6 : 1,
      }).addTo(map)
      all.extend(poly.getBounds())
      if (isFocus) target = poly.getBounds()

      poly.bindTooltip(
        `<strong style="font-size:13px">${r.shortName}</strong><br/><span style="font-size:11px;opacity:.75">${r.speciesCount.toLocaleString()} species recorded · click to explore</span>`,
        { sticky: true, direction: "top", className: "reserve-tooltip" },
      )
      if (!isFocus) poly.on("click", () => router.push(`/reserves/${r.slug}`))
      poly.on("mouseover", () => !isFocus && poly.setStyle({ fillOpacity: 0.5, weight: 3 }))
      poly.on("mouseout", () => !isFocus && poly.setStyle({ fillOpacity: dim ? 0.12 : 0.35, weight: 2 }))

      // A pin at each reserve so they stay findable when zoomed out.
      if (!focus) {
        L.marker(r.center, {
          icon: L.divIcon({
            className: "reserve-label",
            html: `<div class="reserve-pin" style="--c:${r.color}"><span class="dot"></span><span class="name">${r.shortName}</span></div>`,
            iconSize: [0, 0],
            iconAnchor: [9, 9],
          }),
        })
          .on("click", () => router.push(`/reserves/${r.slug}`))
          .addTo(map)
      }
    })

    // The container can still be sizing itself on first paint; re-fit whenever it changes.
    const fit = () => {
      map.invalidateSize()
      map.fitBounds(target ?? all, { padding: focus ? [40, 40] : [70, 70], maxZoom: focus ? 16 : 12 })
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el.current)
    return () => {
      ro.disconnect()
      map.remove()
    }
  }, [reserves, focus, router])

  return <div ref={el} className={className} />
}
