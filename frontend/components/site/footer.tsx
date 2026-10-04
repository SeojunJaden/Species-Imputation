import { formatDataThrough } from "@/lib/site-data"

export function SiteFooter() {
  return (
    <footer className="relative overflow-hidden bg-forest-deep text-parchment/75">
      <div className="topo text-parchment opacity-[0.06]" />
      <div className="relative mx-auto grid max-w-7xl gap-8 px-4 py-12 text-sm sm:px-6 md:grid-cols-3">
        <div>
          <p className="font-display text-lg text-parchment">San Diego Reserve Explorer</p>
          <p className="mt-2 leading-relaxed">
            An independent student project at UC San Diego. Not an official site of the UC
            Natural Reserve System. Many reserves are open by arrangement only, so check access
            before you visit.
          </p>
        </div>
        <div>
          <p className="font-semibold text-parchment">Where the records come from</p>
          <p className="mt-2 leading-relaxed">
            Observations by the{" "}
            <a className="underline decoration-parchment/40 underline-offset-2 hover:text-parchment" href="https://www.inaturalist.org">
              iNaturalist
            </a>{" "}
            community, current to {formatDataThrough()}. Locations hidden for sensitive species
            are left out, and so are records not identified to species.
          </p>
        </div>
        <div>
          <p className="font-semibold text-parchment">Photos & maps</p>
          <p className="mt-2 leading-relaxed">
            Species photos are iNaturalist images under Creative Commons licences, credited on
            each photo. Map tiles © Esri and its data partners. Reserve outlines are approximate.
          </p>
        </div>
      </div>
    </footer>
  )
}
