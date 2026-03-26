export function LoadingLabel({
  label,
  className,
}: {
  label: string;
  className?: string;
}) {
  const classes = className ? `loading-label ${className}` : "loading-label";

  return <span className={classes}>{label}</span>;
}
