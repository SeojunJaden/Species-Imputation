import { Sparkles } from "lucide-react"
import type { Species } from "@/lib/taxa"
import { PredictedMark } from "@/components/site/legend"
import { SpeciesCard } from "@/components/site/species-card"

/**
 * Model predictions. They stay "pending" until reviewed in the field with reserve
 * staff; flip predictionsStatus in data/reserves.json (build_site_data.py) to publish.
 */
export function PredictionsPanel({
  status,
  predicted,
  reserveName,
}: {
  status: "pending" | "published"
  predicted: Species[]
  reserveName: string
}) {
  return (
    <section className="relative overflow-hidden rounded-3xl border-2 border-dashed border-predicted/60 bg-predicted-soft/60 p-6 sm:p-8">
      <Sparkles className="absolute -right-6 -top-6 h-40 w-40 text-predicted/10" aria-hidden />
      <div className="relative flex flex-col gap-6">
        <div className="flex items-start gap-4">
          <PredictedMark className="mt-1 h-9 w-9 [&>svg]:h-5 [&>svg]:w-5" />
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-predicted">Model predictions</p>
            <h2 className="mt-1 font-display text-2xl font-semibold sm:text-3xl">
              Likely here, not yet recorded
            </h2>
            <p className="mt-2 max-w-2xl leading-relaxed text-foreground/80">
              Our model compares {reserveName} with a nearby reserve of similar habitat and flags
              species that are probably present but that nobody has logged here yet.
            </p>
          </div>
        </div>

        {status === "published" && predicted.length ? (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {predicted.map((s) => (
              <SpeciesCard key={s.id} s={s} kind="predicted" />
            ))}
          </div>
        ) : (
          <div className="rounded-2xl border border-predicted/30 bg-card/70 p-5 sm:p-6">
            <p className="font-medium">Being checked in the field</p>
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
              These predictions are being reviewed on the ground with reserve staff. Once checked,
              they will appear here, marked with the dashed ochre outline, so they are never confused
              with species people have actually recorded.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
