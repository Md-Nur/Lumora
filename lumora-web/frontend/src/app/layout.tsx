import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  variable: "--font-jakarta",
  subsets: ["latin"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Lumora VLM Analytics Engine",
  description: "Advanced Autoregressive Multimodal Chest X-Ray Clinical Report Generator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${jakarta.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="antialiased min-h-screen bg-[#f8fafc] text-slate-800 font-sans">
        {children}
      </body>
    </html>
  );
}
