defmodule VemmaEx.CTC do
  @moduledoc """
  Greedy (best-path) CTC decoding, as `quan_transcribe.model.greedy_decode`
  and `QuanTranscribe.decode_units`.

  Unit ids: 0 = blank, 1 = `<unk>`, 2.. = Gemma tokens (`gemma_ids[u - 2]`).
  """

  @blank 0

  @doc "Collapses repeated ids and drops blanks."
  @spec collapse([non_neg_integer()]) :: [non_neg_integer()]
  def collapse(ids) do
    ids
    |> Enum.dedup()
    |> Enum.reject(&(&1 == @blank))
  end

  @doc "Turns per-frame best ids into text."
  @spec decode(VemmaEx.Model.t(), [non_neg_integer()]) :: String.t()
  def decode(model, ids) do
    ids
    |> collapse()
    |> units_to_text(model)
  end

  @doc "Turns collapsed unit ids into text: drops `<unk>`, maps to Gemma ids, detokenizes, trims."
  @spec units_to_text([non_neg_integer()], VemmaEx.Model.t()) :: String.t()
  def units_to_text(units, model) do
    units
    |> Enum.filter(&(&1 >= 2))
    |> Enum.map(&elem(model.units, &1 - 2))
    |> case do
      [] ->
        ""

      gemma_ids ->
        # Python's tokenizer.decode keeps special tokens; so do we.
        {:ok, text} =
          Tokenizers.Tokenizer.decode(model.tokenizer, gemma_ids, skip_special_tokens: false)

        String.trim(text)
    end
  end

  @doc "Index of the last non-blank frame, or `nil`."
  @spec last_emitted([non_neg_integer()]) :: non_neg_integer() | nil
  def last_emitted(ids) do
    ids
    |> Enum.with_index()
    |> Enum.reduce(nil, fn
      {@blank, _}, acc -> acc
      {_, i}, _ -> i
    end)
  end
end
