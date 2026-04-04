import { cn } from "@/lib/utils"

interface SliderProps {
  className?: string
  value?: number[]
  defaultValue?: number[]
  min?: number
  max?: number
  step?: number
  onValueChange?: (value: number[]) => void
  disabled?: boolean
}

function Slider({
  className,
  value,
  defaultValue,
  min = 0,
  max = 100,
  step = 1,
  onValueChange,
  disabled,
}: SliderProps) {
  const currentValue = value?.[0] ?? defaultValue?.[0] ?? min

  return (
    <div className={cn("relative w-full", className)} data-slot="slider">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={currentValue}
        onChange={(e) => {
          const val = parseFloat(e.target.value)
          onValueChange?.([val])
        }}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-muted
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:size-3.5
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:border
          [&::-webkit-slider-thumb]:border-ring
          [&::-webkit-slider-thumb]:bg-white
          [&::-webkit-slider-thumb]:ring-ring/50
          [&::-webkit-slider-thumb]:transition-shadow
          [&::-webkit-slider-thumb]:hover:ring-3
          [&::-webkit-slider-thumb]:active:ring-3
          [&::-moz-range-thumb]:size-3.5
          [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border
          [&::-moz-range-thumb]:border-ring
          [&::-moz-range-thumb]:bg-white
          disabled:opacity-50
          disabled:pointer-events-none"
        disabled={disabled}
      />
    </div>
  )
}

export { Slider }
export type { SliderProps }
