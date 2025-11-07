interface MetricsTableProps {
  metrics: Record<string, number> | null;
}

export function MetricsTable({ metrics }: MetricsTableProps) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <section>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Evaluation Metrics</h2>
        <p style={{ color: '#4b5563' }}>Metrics will appear here after generating a SOAP note.</p>
      </section>
    );
  }

  return (
    <section>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Evaluation Metrics</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left' }}>
            <th style={{ padding: '0.75rem 0.5rem', borderBottom: '1px solid #e5e7eb' }}>Metric</th>
            <th style={{ padding: '0.75rem 0.5rem', borderBottom: '1px solid #e5e7eb' }}>Score</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([name, value]) => (
            <tr key={name}>
              <td style={{ padding: '0.65rem 0.5rem', borderBottom: '1px solid #f3f4f6' }}>{name}</td>
              <td style={{ padding: '0.65rem 0.5rem', borderBottom: '1px solid #f3f4f6' }}>{value.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}


