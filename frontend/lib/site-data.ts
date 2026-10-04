// Species recorded at each reserve, built from iNaturalist by
// cleaning-pipeline/build_site_data.py. Do not edit data/reserves.json by hand.
import raw from "@/data/reserves.json"

import type { Reserve } from "@/lib/taxa"
export * from "@/lib/taxa"

interface SiteData {
  dataThrough: string
  predictionsStatus: "pending" | "published"
  groupOrder: string[]
  reserves: Reserve[]
}

export const site = raw as unknown as SiteData
export const reserves = site.reserves

export function getReserve(slug: string) {
  return reserves.find((r) => r.slug === slug)
}

export function formatDataThrough() {
  return new Date(site.dataThrough + "T12:00:00").toLocaleDateString("en-US", {
    month: "long", day: "numeric", year: "numeric",
  })
}

export function totals() {
  const ids = new Set<number>()
  let observations = 0
  for (const r of reserves) {
    observations += r.stats.observations
    r.species.forEach((s) => ids.add(s.id))
  }
  return { species: ids.size, observations }
}

/** The slice of each reserve the map needs -- keeps the full species lists off the client. */
export function mapReserves() {
  return reserves.map((r) => ({
    slug: r.slug, shortName: r.shortName, color: r.color,
    boundary: r.boundary, center: r.center, speciesCount: r.stats.species,
  }))
}

/** Groups in display order (animals first), with species counts. */
export function orderedGroups(r: Reserve): [string, number][] {
  const order = site.groupOrder
  return Object.entries(r.groups).sort(
    ([a], [b]) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99),
  )
}

/** A few photographed, frequently seen animals to show off a reserve. */
export function highlights(r: Reserve, n = 4) {
  const animals = new Set(["Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta", "Arachnida"])
  const picked: typeof r.species = []
  const seenGroups = new Set<string>()
  for (const s of r.species) {
    if (!s.photo || !animals.has(s.group) || seenGroups.has(s.group)) continue
    picked.push(s)
    seenGroups.add(s.group)
    if (picked.length === n) break
  }
  return picked
}
