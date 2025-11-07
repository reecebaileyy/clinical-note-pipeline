import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Clinical Note Dashboard',
  description: 'Monitor real-time transcriptions and SOAP note summaries.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
