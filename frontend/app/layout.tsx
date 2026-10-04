import type { Metadata, Viewport } from "next"
import { Fraunces, Inter } from "next/font/google"

import "./globals.css"
import { SiteHeader } from "@/components/site/header"
import { SiteFooter } from "@/components/site/footer"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })
const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", axes: ["opsz"] })

export const metadata: Metadata = {
  title: "San Diego Reserve Explorer",
  description:
    "Explore the wildlife and plants recorded at four UC Natural Reserve System reserves managed by UC San Diego.",
}

export const viewport: Viewport = {
  themeColor: "#1f3a2c",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="flex min-h-screen flex-col">
        <SiteHeader />
        <main className="flex-1">{children}</main>
        <SiteFooter />
      </body>
    </html>
  )
}
