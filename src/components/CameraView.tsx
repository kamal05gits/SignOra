import type { MutableRefObject } from "react";
import { CameraOff, Loader2 } from "lucide-react";
import { cn } from "../utils/cn";

export function CameraView({
  videoRef,
  canvasRef,
  isActive,
  isStarting,
  mirror,
  overlayTop,
  overlayBottom,
}: {
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  canvasRef: MutableRefObject<HTMLCanvasElement | null>;
  isActive: boolean;
  isStarting: boolean;
  mirror: boolean;
  overlayTop?: React.ReactNode;
  overlayBottom?: React.ReactNode;
}) {
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-2xl bg-slate-900">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn("h-full w-full object-cover", mirror && "-scale-x-100")}
      />
      <canvas
        ref={canvasRef}
        className={cn("pointer-events-none absolute inset-0 h-full w-full", mirror && "-scale-x-100")}
      />
      {!isActive && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-900/95 text-slate-300">
          {isStarting ? (
            <>
              <Loader2 className="h-8 w-8 animate-spin text-teal-400" />
              <p className="text-sm">Requesting camera access…</p>
            </>
          ) : (
            <>
              <CameraOff className="h-8 w-8 text-slate-500" />
              <p className="text-sm">Camera is off</p>
            </>
          )}
        </div>
      )}
      {overlayTop && <div className="absolute inset-x-0 top-0 p-3">{overlayTop}</div>}
      {overlayBottom && <div className="absolute inset-x-0 bottom-0 p-3">{overlayBottom}</div>}
    </div>
  );
}
