type SoapNote = {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
};

interface SoapNoteCardProps {
  note: SoapNote;
}

const sections: Array<{ key: keyof SoapNote; label: string; accent: string }> = [
  { key: 'subjective', label: 'Subjective', accent: '#0284c7' },
  { key: 'objective', label: 'Objective', accent: '#16a34a' },
  { key: 'assessment', label: 'Assessment', accent: '#7c3aed' },
  { key: 'plan', label: 'Plan', accent: '#ea580c' },
];

export function SoapNoteCard({ note }: SoapNoteCardProps) {
  return (
    <section>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>SOAP Note</h2>
      <div className="grid">
        {sections.map((section) => (
          <div
            key={section.key}
            style={{
              borderLeft: `4px solid ${section.accent}`,
              paddingLeft: '1rem',
            }}
          >
            <h3 style={{ marginBottom: '0.35rem' }}>{section.label}</h3>
            <p style={{ lineHeight: 1.6 }}>{note[section.key] || '—'}</p>
          </div>
        ))}
      </div>
    </section>
  );
}


