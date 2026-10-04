// Labels and helpers shared by server and client components. Kept free of the
// dataset import so client bundles stay small.

export interface Photo {
  url: string
  attribution: string
  license: string
}

export interface Species {
  id: number
  name: string
  common: string | null
  group: string
  observations: number
  /** observations per calendar month, Jan..Dec */
  months: number[]
  lastSeen: string | null
  photo: Photo | null
}

export interface Reserve {
  slug: string
  name: string
  shortName: string
  color: string
  setting: string
  habitats: string[]
  description: string
  /** [lat, lon] */
  boundary: [number, number][]
  center: [number, number]
  areaHa: number
  stats: { observations: number; species: number; observers: number; firstYear: number }
  groups: Record<string, number>
  species: Species[]
  /** Model predictions. Empty until reviewed with reserve staff (predictionsStatus). */
  predicted: Species[]
}

export const ANIMAL_GROUPS = new Set([
  "Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta", "Arachnida",
  "Mollusca", "Actinopterygii", "Animalia",
])

/** Plain-language names for iNaturalist's iconic taxon groups. */
export const GROUP_LABEL: Record<string, string> = {
  Aves: "Birds",
  Mammalia: "Mammals",
  Reptilia: "Reptiles",
  Amphibia: "Amphibians",
  Insecta: "Insects",
  Arachnida: "Spiders & kin",
  Mollusca: "Snails & molluscs",
  Actinopterygii: "Fish",
  Animalia: "Other animals",
  Plantae: "Plants",
  Fungi: "Fungi & lichens",
  Chromista: "Kelp & algae",
  Protozoa: "Slime molds",
  Unknown: "Other",
}

export function groupLabel(g: string) {
  return GROUP_LABEL[g] ?? g
}

/** How often a species turns up, in words a visitor can use. */
export function frequency(observations: number) {
  if (observations >= 20) return { label: "Frequently seen", level: 3 }
  if (observations >= 5) return { label: "Occasionally seen", level: 2 }
  return { label: "Rarely recorded", level: 1 }
}

export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
