import { ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { MONTHS, frequency, groupLabel, type Species } from "@/lib/taxa"
import { GroupIcon } from "@/components/site/group-icon"
import { ObservedMark, PredictedMark } from "@/components/site/legend"

/** Twelve bars: when in the year this species has been recorded here. */
export function MonthBars({ months, kind }: { months: number[]; kind: "observed" | "predicted" }) {
  const max = Math.max(1, ...months)
  return (
    <div className="flex items-end gap-[3px]" aria-label="Records by month">
      {months.map((n, i) => (
        <div key={i} className="flex flex-1 flex-col items-center gap-1">
          <div className="flex h-7 w-full items-end overflow-hidden rounded-sm bg-muted">
            <div
              className={cn("w-full rounded-sm", kind === "observed" ? "bg-observed" : "bg-predicted")}
              style={{ height: `${n ? Math.max(14, (n / max) * 100) : 0}%`, opacity: n ? 0.85 : 0 }}
              title={`${MONTHS[i]}: ${n} record${n === 1 ? "" : "s"}`}
            />
          </div>
          <span className="text-[9px] leading-none text-muted-foreground">{MONTHS[i][0]}</span>
        </div>
      ))}
    </div>
  )
}

export function SpeciesCard({ s, kind = "observed" }: { s: Species; kind?: "observed" | "predicted" }) {
  const freq = frequency(s.observations)
  const observed = kind === "observed"
  return (
    <article
      className={cn(
        "group flex flex-col overflow-hidden rounded-2xl bg-card shadow-sm transition hover:-translate-y-0.5 hover:shadow-lg",
        observed ? "border border-border" : "border-2 border-dashed border-predicted/70",
      )}
    >
      <div className="relative aspect-[4/3] overflow-hidden bg-secondary">
        {s.photo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={s.photo.url}
            alt={s.common ?? s.name}
            loading="lazy"
            className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
          />
        ) : (
          <div className="paper flex h-full w-full items-center justify-center">
            <GroupIcon group={s.group} className="h-12 w-12 text-muted-foreground/40" />
          </div>
        )}
        <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full bg-card/90 py-1 pl-1 pr-2.5 text-xs font-medium shadow-sm backdrop-blur">
          {observed ? <ObservedMark /> : <PredictedMark />}
          {observed ? "Seen here" : "Predicted"}
        </div>
        {s.photo && (
          <p className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/60 to-transparent px-3 pb-1.5 pt-6 text-[10px] text-white/85">
            {s.photo.attribution}
          </p>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div>
          <h3 className="font-display text-lg font-semibold leading-snug">
            {s.common ?? <i>{s.name}</i>}
          </h3>
          {s.common && <p className="text-sm italic text-muted-foreground">{s.name}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="inline-flex items-center gap-1 rounded-full bg-secondary px-2 py-0.5 text-secondary-foreground">
            <GroupIcon group={s.group} className="h-3.5 w-3.5" />
            {groupLabel(s.group)}
          </span>
          {observed && (
            <span className="inline-flex items-center gap-1 text-muted-foreground" title={`${s.observations} records`}>
              <span className="flex gap-0.5" aria-hidden>
                {[1, 2, 3].map((l) => (
                  <span key={l} className={cn("h-1.5 w-1.5 rounded-full", l <= freq.level ? "bg-observed" : "bg-border")} />
                ))}
              </span>
              {freq.label}
            </span>
          )}
        </div>
        <div className="mt-auto space-y-2">
          <MonthBars months={s.months} kind={kind} />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {s.observations} record{s.observations === 1 ? "" : "s"}
              {s.lastSeen && ` · last ${new Date(s.lastSeen + "T12:00:00").toLocaleDateString("en-US", { month: "short", year: "numeric" })}`}
            </span>
            <a
              href={`https://www.inaturalist.org/taxa/${s.id}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
            >
              iNat <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </div>
    </article>
  )
}
