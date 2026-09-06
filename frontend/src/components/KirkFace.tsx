interface KirkFaceProps {
  size?: "small" | "large";
}

export default function KirkFace({
  size = "small",
}: KirkFaceProps) {
  return (
    <div
      className={`kirk-face kirk-face-${size}`}
      aria-hidden="true"
    >
      <div className="kirk-hair">
        <div className="kirk-part" />
      </div>

      <div className="kirk-ear kirk-ear-left" />
      <div className="kirk-ear kirk-ear-right" />

      <div className="kirk-brow kirk-brow-left" />
      <div className="kirk-brow kirk-brow-right" />

      <div className="kirk-eye kirk-eye-left">
        <div className="kirk-pupil" />
      </div>

      <div className="kirk-eye kirk-eye-right">
        <div className="kirk-pupil" />
      </div>

      <div className="kirk-nose" />

      <div className="kirk-mouth" />

      <div className="kirk-chin" />
    </div>
  );
}
