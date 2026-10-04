"use client"

import { useMemo, useState } from "react"
import { Search, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { ANIMAL_GROUPS, MONTHS, groupLabel, type Species } from "@/lib/taxa"
import { GroupIcon } from "@/components/site/group-icon"
import { SpeciesCard } from "@/components/site/species-card"

type Sort = "seen" | "name" | "recent"
const PAGE = 24

interface Props {
  species: Species[]
  groups: [string, number][]
}

export function SpeciesExplorer({ species, groups }: Props) {
  const [group, setGroup] = useState<string>("animals")
  const [query, setQuery] = useState("")
  const [month, setMonth] = useState<number | null>(null)
  const [sort, setSort] = useState<Sort>("seen")
  const [shown, setShown] = useState(PAGE)

  const animalCount = groups.filter(([g]) => ANIMAL_GROUPS.has(g)).reduce((n, [, c]) => n + c, 0)

  const list = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out = species.filter((s) => {
      if (group === "animals" && !ANIMAL_GROUPS.has(s.group)) return false
      if (group !== "animals" && group !== "all" && s.group !== group) return false
      if (month !== null && !s.months[month]) return false
      if (q && !s.name.toLowerCase().includes(q) && !(s.common ?? "").toLowerCase().includes(q)) return false
      return true
    })
    const byName = (a: Species, b: Species) => (a.common ?? a.name).localeCompare(b.common ?? b.name)
    if (sort === "name") out.sort(byName)
    else if (sort === "recent") out.sort((a, b) => (b.lastSeen ?? "").localeCompare(a.lastSeen ?? ""))
    else if (month !== null) out.sort((a, b) => b.months[month] - a.months[month] || byName(a, b))
    else out.sort((a, b) => b.observations - a.observations || byName(a, b))
    return out
  }, [species, group, query, month, sort])

  const reset = () => setShown(PAGE)

  return (
    <div className="space-y-6">
      {/* Group chips: animals first */}
      <div className="-mx-4 overflow-x-auto px-4 pb-1 sm:mx-0 sm:px-0">
        <div className="flex w-max gap-2 sm:w-auto sm:flex-wrap">
          <Chip active={group === "animals"} onClick={() => { setGroup("animals"); reset() }} count={animalCount}>
            All animals
          </Chip>
          {groups.map(([g, n]) => (
            <Chip key={g} active={group === g} onClick={() => { setGroup(g); reset() }} count={n}>
              <GroupIcon group={g} className="h-3.5 w-3.5" />
              {groupLabel(g)}
            </Chip>
          ))}
          <Chip active={group === "all"} onClick={() => { setGroup("all"); reset() }} count={species.length}>
            Everything
          </Chip>
        </div>
      </div>

      {/* Search, season, sort */}
      <div className="flex flex-col gap-3 rounded-2xl border bg-card p-3 shadow-sm lg:flex-row lg:items-center">
        <label className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => { setQuery(e.target.value); reset() }}
            placeholder="Search by common or scientific name"
            className="h-10 w-full rounded-xl border bg-background pl-9 pr-9 text-sm outline-none ring-ring/30 focus:ring-2"
          />
          {query && (
            <button onClick={() => setQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground" aria-label="Clear search">
              <X className="h-4 w-4" />
            </button>
          )}
        </label>
        <div className="flex items-center gap-1 overflow-x-auto" role="group" aria-label="Season">
          <span className="mr-1 shrink-0 text-xs font-medium text-muted-foreground">Season</span>
          <MonthButton active={month === null} onClick={() => { setMonth(null); reset() }}>All</MonthButton>
          {MONTHS.map((m, i) => (
            <MonthButton key={m} active={month === i} onClick={() => { setMonth(i); reset() }}>
              {m}
            </MonthButton>
          ))}
        </div>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as Sort)}
          className="h-10 rounded-xl border bg-background px-3 text-sm"
          aria-label="Sort"
        >
          <option value="seen">Most often seen</option>
          <option value="recent">Recently seen</option>
          <option value="name">A–Z</option>
        </select>
      </div>

      <p className="text-sm text-muted-foreground">
        {list.length.toLocaleString()} species
        {month !== null && ` recorded in ${MONTHS[month]}`}
      </p>

      {list.length ? (
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {list.slice(0, shown).map((s) => (
            <SpeciesCard key={s.id} s={s} />
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed p-10 text-center text-muted-foreground">
          No species match. Try another group, month or search.
        </div>
      )}

      {shown < list.length && (
        <div className="flex justify-center">
          <button
            onClick={() => setShown((n) => n + PAGE)}
            className="rounded-full border border-primary/30 bg-card px-6 py-2.5 text-sm font-medium text-primary shadow-sm transition hover:bg-primary hover:text-primary-foreground"
          >
            Show more ({(list.length - shown).toLocaleString()} left)
          </button>
        </div>
      )}
    </div>
  )
}

function Chip({ active, onClick, count, children }: { active: boolean; onClick: () => void; count: number; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-sm transition",
        active ? "border-primary bg-primary text-primary-foreground shadow-sm" : "bg-card hover:border-primary/40",
      )}
    >
      {children}
      <span className={cn("text-xs", active ? "text-primary-foreground/70" : "text-muted-foreground")}>{count}</span>
    </button>
  )
}

function MonthButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "shrink-0 rounded-lg px-2 py-1 text-xs transition",
        active ? "bg-observed text-white" : "text-muted-foreground hover:bg-secondary",
      )}
    >
      {children}
    </button>
  )
}
