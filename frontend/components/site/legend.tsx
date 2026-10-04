import { Binoculars, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"

/**
 * The one visual distinction the site must never blur: species people have
 * actually recorded (sage, solid) versus species our model expects (ochre, dashed).
 */
export function ObservedMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 w-5 items-center justify-center rounded-full bg-observed text-white",
        className,
      )}
      aria-hidden
    >
      <Binoculars className="h-3 w-3" />
    </span>
  )
}

export function PredictedMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 w-5 items-center justify-center rounded-full border-2 border-dashed border-predicted bg-predicted-soft text-predicted",
        className,
      )}
      aria-hidden
    >
      <Sparkles className="h-3 w-3" />
    </span>
  )
}

export function Legend({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border bg-card/90 px-4 py-3 text-sm shadow-sm backdrop-blur",
        className,
      )}
      role="note"
      aria-label="Legend"
    >
      {!compact && (
        <span className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Legend
        </span>
      )}
      <span className="flex items-center gap-2">
        <ObservedMark />
        <span>
          <span className="font-medium text-observed">Seen on iNaturalist</span>
          {!compact && <span className="text-muted-foreground"> · recorded here by visitors</span>}
        </span>
      </span>
      <span className="flex items-center gap-2">
        <PredictedMark />
        <span>
          <span className="font-medium text-predicted">Model prediction</span>
          {!compact && (
            <span className="text-muted-foreground"> · likely here, not yet recorded</span>
          )}
        </span>
      </span>
    </div>
  )
}
