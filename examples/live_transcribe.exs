# Live transcription in a Phoenix LiveView, from the browser microphone, from
# screen / tab audio, or both at once (two transcripts side by side).
#
#     VEMMA_MODEL_DIR=/path/to/QUAN-Transcribe-1.0/onnx elixir examples/live_transcribe.exs
#
# then open http://localhost:4000, pick a source and press Start. Without
# VEMMA_MODEL_DIR the model is downloaded (1.4 GB) to vemma_ex's priv/models.
#
#     elixir examples/live_transcribe.exs --check path/to/16k.wav
#
# feeds a WAV through the same "start"/"chunk"/"stop" events the browser sends,
# as two sources at once, in 0.48 s pieces, and checks the text of the Live path.
#
# Browser: the microphone comes from getUserMedia (echo cancellation and noise
# suppression on), screen audio from getDisplayMedia (video is required by
# browsers and stopped at once; only the audio track is kept). Tab / system
# audio capture is Chrome and Edge only. Each source goes through its own
# AudioWorklet that averages the channels to mono, box-filters and decimates to
# 16 kHz f32, and posts 7680-sample (0.48 s) chunks that the hook pushes as base64.
#
# Server: one VemmaEx.Live session per source per LiveView process, run with
# start_async (one decode per source at a time). Chunks that arrive during a
# decode are coalesced into the next one, keeping at most the newest 10 s.
Mix.install([
  {:vemma_ex, path: Path.expand("..", __DIR__)},
  {:phoenix_playground, "~> 0.1.9"}
])

defmodule LiveTranscribe.Page do
  use Phoenix.LiveView

  require Logger

  @sources ["mic", "screen"]
  @labels %{"mic" => "Microphone", "screen" => "Screen / tab audio"}
  # Coalesced audio waiting for the decoder, as f32 bytes; older audio beyond this is dropped.
  @max_pending 10 * 16_000 * 4

  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "vemma_ex live", sources: %{})}
  end

  def handle_event("start", %{"source" => source}, socket) when source in @sources do
    state =
      socket.assigns.sources
      |> Map.get(source, %{session: nil, lines: [], dropped: 0})
      |> then(&%{&1 | session: &1.session || VemmaEx.Live.new(:default)})
      |> Map.merge(%{partial: "", pending: <<>>, busy: false, stopping: false, recording: true})

    {:noreply, put_source(socket, source, state)}
  end

  def handle_event("chunk", %{"source" => source, "data" => data}, socket) do
    {Map.fetch(socket.assigns.sources, source), Base.decode64(data)}
    |> case do
      {{:ok, %{recording: true} = state}, {:ok, bytes}} when rem(byte_size(bytes), 4) == 0 ->
        {:noreply, socket |> put_source(source, enqueue(state, bytes)) |> run(source)}

      _ ->
        {:noreply, socket}
    end
  end

  def handle_event("stop", %{"source" => source}, socket) do
    socket.assigns.sources
    |> Map.fetch(source)
    |> case do
      {:ok, state} ->
        {:noreply,
         socket
         |> put_source(source, %{state | recording: false, stopping: true})
         |> run(source)}

      :error ->
        {:noreply, socket}
    end
  end

  def handle_async({:accept, source}, {:ok, {finals, session}}, socket) do
    state = socket.assigns.sources[source]

    {:noreply,
     socket
     |> put_source(source, %{
       state
       | session: session,
         lines: state.lines ++ finals,
         partial: session.partial,
         busy: false
     })
     |> run(source)}
  end

  def handle_async({:flush, source}, {:ok, {text, session}}, socket) do
    state = socket.assigns.sources[source]

    {:noreply,
     socket
     |> put_source(source, %{
       state
       | session: session,
         lines: add_line(state.lines, text),
         partial: "",
         busy: false
     })
     |> run(source)}
  end

  def handle_async({_, source} = name, {:exit, reason}, socket) do
    Logger.error("live_transcribe: #{inspect(name)} failed: #{inspect(reason)}")
    {:noreply, socket |> put_source(source, %{socket.assigns.sources[source] | busy: false})}
  end

  defp put_source(socket, source, state),
    do: assign(socket, sources: Map.put(socket.assigns.sources, source, state))

  defp add_line(lines, ""), do: lines
  defp add_line(lines, text), do: lines ++ [text]

  defp enqueue(state, bytes) do
    pending = state.pending <> bytes
    over = max(byte_size(pending) - @max_pending, 0)

    %{
      state
      | pending: binary_part(pending, over, byte_size(pending) - over),
        dropped: state.dropped + div(over, 4)
    }
  end

  # One decode per source at a time; whatever arrived meanwhile goes into the next one.
  defp run(socket, source) do
    state = socket.assigns.sources[source]
    session = state.session

    {state.busy, state.pending, state.stopping}
    |> case do
      {true, _, _} ->
        socket

      {false, <<>>, true} ->
        socket
        |> put_source(source, %{state | busy: true, stopping: false})
        |> start_async({:flush, source}, fn -> VemmaEx.Live.flush(session) end)

      {false, <<>>, false} ->
        socket

      {false, pending, _} ->
        socket
        |> put_source(source, %{state | busy: true, pending: <<>>})
        |> start_async({:accept, source}, fn -> VemmaEx.Live.accept(session, pending) end)
    end
  end

  defp status(state) do
    [
      if(state.recording, do: "listening", else: "stopped"),
      if(state.busy, do: "decoding"),
      if(state.dropped > 0,
        do: "dropped #{Float.round(state.dropped / 16_000, 1)} s while behind"
      )
    ]
    |> Enum.reject(&is_nil/1)
    |> Enum.join(" · ")
  end

  def render(assigns) do
    assigns = assign(assigns, labels: @labels, order: @sources)

    ~H"""
    <style>
      :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
      main { max-width: 72rem; margin: 2rem auto; padding: 0 1rem; }
      button { font-size: 1rem; padding: .5rem 1.2rem; margin-right: .5rem; }
      fieldset { border: 0; padding: 0; margin: 0 0 1rem; display: flex; gap: 1.2rem; flex-wrap: wrap; }
      .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 2rem; }
      ol { padding-left: 1.5rem; line-height: 1.6; }
      .partial { opacity: .6; font-style: italic; min-height: 1.6em; }
      .status { font-size: .85rem; opacity: .7; }
      #capture-status { display: block; margin-top: .8rem; color: #c0392b; }
    </style>
    <main>
      <h1>vemma_ex live transcription</h1>
      <div id="capture" phx-hook="Capture" phx-update="ignore">
        <fieldset>
          <label><input type="radio" name="source" value="mic" checked /> Microphone</label>
          <label><input type="radio" name="source" value="screen" /> Screen / tab audio</label>
          <label><input type="radio" name="source" value="both" /> Both</label>
        </fieldset>
        <button id="start">Start</button>
        <button id="stop" disabled>Stop</button>
        <span id="capture-status"></span>
      </div>
      <div class="columns">
        <section :for={source <- @order} :if={@sources[source]} id={"source-#{source}"}>
          <h2>{@labels[source]}</h2>
          <p class="status">{status(@sources[source])}</p>
          <ol id={"lines-#{source}"}>
            <li :for={line <- @sources[source].lines}>{line}</li>
          </ol>
          <p id={"partial-#{source}"} class="partial">{@sources[source].partial}</p>
        </section>
      </div>
    </main>
    <script phx-update="ignore" id="capture-hook">
      const NO_AUDIO = "The shared screen has no audio. Share a browser tab, or the whole screen, " +
        "with \"Share audio\" ticked (Chrome or Edge; Firefox and Safari cannot capture tab or system audio)."

      window.hooks.Capture = {
        mounted() {
          this.captures = {}
          this.startBtn = this.el.querySelector("#start")
          this.stopBtn = this.el.querySelector("#stop")
          this.status = this.el.querySelector("#capture-status")
          this.startBtn.addEventListener("click", () => this.start())
          this.stopBtn.addEventListener("click", () => this.stopAll())
        },
        destroyed() { Object.keys(this.captures).forEach(s => this.release(s)) },
        toggle(on) {
          this.startBtn.disabled = on; this.stopBtn.disabled = !on
          this.el.querySelectorAll("input").forEach(i => i.disabled = on)
        },
        async start() {
          const choice = this.el.querySelector("input[name=source]:checked").value
          const sources = choice === "both" ? ["mic", "screen"] : [choice]
          this.status.textContent = ""
          this.toggle(true)
          for (const source of sources) {
            try {
              await this.open(source)
            } catch (e) {
              this.release(source)
              this.status.textContent = e.message === "no-audio" ? NO_AUDIO : `${source}: ${e.message || e}`
            }
          }
          if (Object.keys(this.captures).length === 0) this.toggle(false)
        },
        async open(source) {
          const stream = source === "mic" ? await micStream() : await screenStream()
          const ctx = new AudioContext()
          const capture = this.captures[source] = { stream, ctx, node: null, stopping: false }
          const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }))
          await ctx.audioWorklet.addModule(url)
          URL.revokeObjectURL(url)
          // The context runs at the device rate; the worklet resamples to 16 kHz
          // (Firefox cannot connect a stream to a context at another rate).
          const node = capture.node =
            new AudioWorkletNode(ctx, "vemma-downsampler", { numberOfInputs: 1, numberOfOutputs: 0 })
          node.port.onmessage = ({ data }) => {
            if (data.byteLength > 0) this.pushEvent("chunk", { source, data: toBase64(data) })
            if (capture.stopping) { this.release(source); this.pushEvent("stop", { source }) }
          }
          // The browser's own "Stop sharing" bar, or an unplugged mic, ends the track.
          stream.getAudioTracks().forEach(t => t.addEventListener("ended", () => this.stop(source)))
          ctx.createMediaStreamSource(stream).connect(node)
          this.pushEvent("start", { source })
        },
        // Asks the worklet for its last partial chunk; the reply finishes the source.
        stop(source) {
          const capture = this.captures[source]
          if (!capture || capture.stopping) return
          capture.stopping = true
          capture.node.port.postMessage("flush")
        },
        stopAll() { Object.keys(this.captures).forEach(s => this.stop(s)) },
        release(source) {
          const capture = this.captures[source]
          delete this.captures[source]
          if (capture) {
            capture.stream.getTracks().forEach(t => t.stop())
            capture.ctx.close()
          }
          if (Object.keys(this.captures).length === 0) this.toggle(false)
        }
      }

      function micStream() {
        return navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
        })
      }

      async function screenStream() {
        const display = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: {
            echoCancellation: false, noiseSuppression: false, autoGainControl: false,
            suppressLocalAudioPlayback: false
          },
          systemAudio: "include",
          selfBrowserSurface: "exclude"
        })
        display.getVideoTracks().forEach(t => t.stop())
        const audio = display.getAudioTracks()
        if (audio.length === 0) throw new Error("no-audio")
        return new MediaStream(audio)
      }

      function toBase64(buffer) {
        const bytes = new Uint8Array(buffer)
        let s = ""
        for (let i = 0; i < bytes.length; i += 0x8000) {
          s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
        }
        return btoa(s)
      }

      // Mono downmix, then a box-filter decimation to 16 kHz: each output sample
      // averages the input samples of its 1/16000 s span (a cheap anti-alias filter).
      const WORKLET = `
        const RATE = 16000, CHUNK = 7680
        class Downsampler extends AudioWorkletProcessor {
          constructor() {
            super()
            this.step = sampleRate / RATE
            this.pos = 0; this.sum = 0; this.n = 0; this.last = 0
            this.out = new Float32Array(CHUNK); this.k = 0
            this.port.onmessage = () => this.send(this.k)
          }
          send(k) {
            const chunk = this.out.slice(0, k)
            this.port.postMessage(chunk.buffer, [chunk.buffer])
            this.k = 0
          }
          process(inputs) {
            const input = inputs[0]
            if (!input || input.length === 0) return true
            for (let i = 0; i < input[0].length; i++) {
              let s = 0
              for (let c = 0; c < input.length; c++) s += input[c][i]
              this.sum += s / input.length; this.n++; this.pos++
              while (this.pos >= this.step) {
                this.pos -= this.step
                this.last = this.n > 0 ? this.sum / this.n : this.last
                this.sum = 0; this.n = 0
                this.out[this.k++] = this.last
                if (this.k === CHUNK) this.send(CHUNK)
              }
            }
            return true
          }
        }
        registerProcessor("vemma-downsampler", Downsampler)
      `
    </script>
    """
  end
end

# PhoenixPlayground's code reloader evaluates this whole file again on each
# request; only the first evaluation loads the model and starts anything.
:live_transcribe
|> Application.get_env(:started, false)
|> case do
  true ->
    :reloaded

  false ->
    Application.put_env(:live_transcribe, :started, true)

    model_opts =
      "VEMMA_MODEL_DIR"
      |> System.get_env()
      |> case do
        nil -> []
        dir -> [path: dir]
      end

    IO.puts("loading the model...")
    VemmaEx.Model.Registry.load!(:default, model_opts)

    System.argv()
    |> case do
      ["--check", wav] ->
        Logger.configure(level: :warning)
        Application.put_env(:live_transcribe, :wav, wav)
        ExUnit.start()

        defmodule LiveTranscribe.Check do
          use ExUnit.Case
          use PhoenixPlayground.Test, live: LiveTranscribe.Page

          @chunk 7680 * 4

          @tag timeout: 600_000
          test "a WAV fed as 0.48 s chunks from two sources comes out as text" do
            {:ok, samples} =
              VemmaEx.Audio.read_wav(Application.fetch_env!(:live_transcribe, :wav))

            {:ok, view, html} = live(build_conn(), "/")
            assert html =~ "Screen / tab audio"

            bytes = Nx.to_binary(samples)
            tail = rem(byte_size(bytes), @chunk)
            chunks = for <<chunk::binary-size(@chunk) <- bytes>>, do: chunk
            chunks = chunks ++ [binary_part(bytes, byte_size(bytes) - tail, tail)]

            Enum.each(["mic", "screen"], &render_hook(view, "start", %{"source" => &1}))

            # At 4x real time, so chunks arrive while a decode runs and get coalesced.
            Enum.each(chunks, fn chunk ->
              Enum.each(["mic", "screen"], fn source ->
                render_hook(view, "chunk", %{"source" => source, "data" => Base.encode64(chunk)})
              end)

              Process.sleep(120)
            end)

            Enum.each(["mic", "screen"], &render_hook(view, "stop", %{"source" => &1}))

            html = wait_idle(view, 600)
            refute html =~ "dropped"
            {:ok, full} = VemmaEx.transcribe(:default, samples)

            for source <- ["mic", "screen"] do
              text = lines(html, source)
              IO.puts("\n#{source}: #{text}")
              assert String.jaro_distance(text, full) > 0.9
            end

            IO.puts("full:   #{full}")
          end

          # Polls until no source is decoding and both have a line.
          defp wait_idle(view, tries) do
            render_async(view, 60_000)
            html = render(view)

            (html =~ "decoding" or lines(html, "mic") == "" or lines(html, "screen") == "")
            |> case do
              true when tries > 0 ->
                Process.sleep(100)
                wait_idle(view, tries - 1)

              _ ->
                html
            end
          end

          defp lines(html, source) do
            ~r{<ol id="lines-#{source}">(.*?)</ol>}s
            |> Regex.run(html, capture: :all_but_first)
            |> case do
              [list] ->
                ~r{<li[^>]*>(.*?)</li>}s
                |> Regex.scan(list, capture: :all_but_first)
                |> Enum.map_join(" ", &(&1 |> hd() |> String.trim()))

              nil ->
                ""
            end
          end
        end

      _ ->
        PhoenixPlayground.start(live: LiveTranscribe.Page, port: 4000, open_browser: false)
        IO.puts("open http://localhost:4000")
        Process.sleep(:infinity)
    end
end
