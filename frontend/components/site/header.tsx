import Link from "next/link"
import { Compass } from "lucide-react"
import { reserves } from "@/lib/site-data"

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-[1000] border-b border-white/10 bg-forest-deep/95 text-parchment backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" className="group flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-full border border-parchment/30 bg-parchment/10 transition group-hover:rotate-45">
            <Compass className="h-5 w-5" />
          </span>
          <span className="leading-tight">
            <span className="block font-display text-lg font-semibold tracking-tight">
              San Diego Reserve Explorer
            </span>
            <span className="hidden text-[11px] uppercase tracking-[0.18em] text-parchment/60 sm:block">
              UC Natural Reserve System · UC San Diego
            </span>
          </span>
        </Link>
        <nav className="hidden items-center gap-1 md:flex" aria-label="Reserves">
          {reserves.map((r) => (
            <Link
              key={r.slug}
              href={`/reserves/${r.slug}`}
              className="flex items-center gap-2 rounded-full px-3 py-1.5 text-sm text-parchment/80 transition hover:bg-parchment/10 hover:text-parchment"
            >
              <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
              {r.shortName}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}
