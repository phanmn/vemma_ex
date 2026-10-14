defmodule VemmaEx.CTCTest do
  use ExUnit.Case, async: true

  test "collapse merges repeats and drops blanks" do
    assert VemmaEx.CTC.collapse([0, 5, 5, 0, 5, 7, 7, 1, 0]) == [5, 5, 7, 1]
    assert VemmaEx.CTC.collapse([0, 0, 0]) == []
    assert VemmaEx.CTC.collapse([]) == []
  end

  test "last_emitted is the last non-blank frame" do
    assert VemmaEx.CTC.last_emitted([0, 3, 0, 4, 0, 0]) == 3
    assert VemmaEx.CTC.last_emitted([0, 0]) == nil
  end
end
