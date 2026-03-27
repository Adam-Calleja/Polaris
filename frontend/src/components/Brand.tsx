import { Link } from "react-router-dom";
import { useAppState } from "../app/state";

export function Brand() {
  const { dispatch } = useAppState();

  return (
    <Link
      aria-label="Polaris"
      className="brand"
      onClick={() => {
        dispatch({ type: "set-drawer-open", open: false });
      }}
      to="/assistant"
    >
      <span className="brand__icon">✳</span>
      <span className="brand__text">Polaris</span>
    </Link>
  );
}
