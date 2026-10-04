import {
  Bird, Bug, Droplets, Fish, Flower2, Leaf, PawPrint, Shell, Snail, Sprout, Turtle, Worm,
  type LucideIcon,
} from "lucide-react"

const ICONS: Record<string, LucideIcon> = {
  Aves: Bird,
  Mammalia: PawPrint,
  Reptilia: Turtle,
  Amphibia: Droplets,
  Insecta: Bug,
  Arachnida: Bug,
  Mollusca: Snail,
  Actinopterygii: Fish,
  Animalia: Shell,
  Plantae: Leaf,
  Fungi: Sprout,
  Chromista: Flower2,
  Protozoa: Worm,
}

export function GroupIcon({ group, className }: { group: string; className?: string }) {
  const Icon = ICONS[group] ?? Leaf
  return <Icon className={className} aria-hidden />
}
