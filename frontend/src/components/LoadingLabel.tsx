export function LoadingLabel({
  label,
  className,
}: {
  label: string;
  className?: string;
}) {
  const classes = className ? `loading-label ${className}` : "loading-label";

  return (
    <span className={classes}>
      <span>{label}</span>
      <span aria-hidden="true" className="loading-dots">
        <span className="loading-dots__dot">.</span>
        <span className="loading-dots__dot">.</span>
        <span className="loading-dots__dot">.</span>
      </span>
    </span>
  );
}
