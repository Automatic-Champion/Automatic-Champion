import { useEffect, useState } from "react";
import { Loader2, Save } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { saveSquad } from "../../api/client";
import type {
  SavedSquadResponse,
  SquadGenerateResponse,
} from "../../api/types";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  squad: SquadGenerateResponse;
  onSaved: (saved: SavedSquadResponse) => void;
}

export default function SaveSquadDialog({
  open,
  onOpenChange,
  squad,
  onSaved,
}: Props) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setName("");
      setError(null);
      setIsSaving(false);
    }
  }, [open]);

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Please enter a name for this squad.");
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const saved = await saveSquad(trimmed, squad);
      onSaved(saved);
      onOpenChange(false);
    } catch (err) {
      if (err instanceof TypeError && err.message === "Failed to fetch") {
        setError("Cannot connect to the backend. Is the server running?");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to save squad.");
      }
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Save className="size-4 text-amber-500" />
            Save squad
          </DialogTitle>
          <DialogDescription>
            Give this squad a name. You can load it later from the Lineup
            Advisor.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2">
          <Label htmlFor="squad-name">Squad name</Label>
          <Input
            id="squad-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Salah-Haaland double-up"
            maxLength={50}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && !isSaving) {
                e.preventDefault();
                handleSave();
              }
            }}
          />
          {error && (
            <p className="text-xs text-destructive" role="alert">
              {error}
            </p>
          )}
        </div>

        <div className="-mx-4 -mb-4 flex flex-col-reverse gap-2 rounded-b-xl border-t bg-muted/50 p-4 sm:flex-row sm:justify-end">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving && <Loader2 className="size-4 animate-spin" />}
            Save
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
