import { Link } from "react-router-dom";

export function Brand() {
  return (
    <Link className="brand" to="/assistant">
      <span className="brand__icon">✳</span>
      <span className="brand__text">Polaris</span>
    </Link>
  );
}
