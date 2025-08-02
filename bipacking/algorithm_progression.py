import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict


def labels_collide(label1, label2, pad=1.0):
    """
    Returns True if label1 and label2 bounding boxes overlap (in display coords).
    'pad' adds extra space around them so they're not too tight.
    """
    bbox1 = label1.get_window_extent()
    bbox2 = label2.get_window_extent()
    # Expand each box by 'pad' factor
    bbox1 = bbox1.expanded(1 + pad / bbox1.width, 1 + pad / bbox1.height)
    bbox2 = bbox2.expanded(1 + pad / bbox2.width, 1 + pad / bbox2.height)
    return bbox1.fully_overlaps(bbox2)


def place_label_avoiding_collision(ax, x, y, text, **annot_kwargs):
    """
    Creates a text label at data coords (x,y). If it collides with an existing
    label in 'ax', we nudge it upward until no overlap.
    Returns the created Text object.
    """
    lbl = ax.text(x, y, text, **annot_kwargs)

    # Force a draw so Matplotlib computes bounding boxes
    plt.draw()

    # Compare bounding boxes against existing text objects
    others = [
        child for child in ax.get_children()
        if isinstance(child, plt.Text) and child is not lbl
    ]

    # We'll nudge this label upward by a fraction of the Y range until no collision
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    nudge_dy = 0.01 * y_range  # 1% of the Y range

    while True:
        collided = False
        for other_lbl in others:
            if labels_collide(lbl, other_lbl):
                x0, y0 = lbl.get_position()
                lbl.set_position((x0, y0 + nudge_dy))
                plt.draw()
                collided = True
        if not collided:
            break

    return lbl


def plot_same_color_labels(filename="Algorithm progression.txt",
                           output_png="progress.png"):
    """
    1) Parses "[Thread X] best score so far: Y" in file order.
    2) Plots each thread's (local_step, score) with a line (no marker). 
       - Captures each line's color in thread_color[t_id].
    3) Identifies repeated scores across multiple threads and their earliest occurrence.
    4) Labels:
       - The earliest occurrence of each repeated score with the color of the thread that found it earliest.
       - The final point of each thread with that thread's color.
    5) Uses bounding-box checks to nudge labels upward if they collide.
    6) Formats the Y-axis to 3 decimals.
    """

    # Regex for lines: [Thread 0] best score so far: 1.2345
    pattern = re.compile(
        r'^\[Thread\s+(\d+)\].*?best score so far:\s+([-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    )

    # We'll track the order in which lines appear in the file
    global_line_index = 0
    # For each thread, a local step
    thread_local_step = defaultdict(int)
    # We store (g_idx, thread_id, local_step, score)
    all_points = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                thr_str, score_str = match.groups()
                thr_id = int(thr_str)
                score_val = float(score_str)

                # Increment local step for this thread
                thread_local_step[thr_id] += 1
                local_step = thread_local_step[thr_id]

                # Save
                all_points.append(
                    (global_line_index, thr_id, local_step, score_val))
                global_line_index += 1

    # Identify repeated scores: those that appear in >= 2 distinct threads
    score_to_threads = defaultdict(set)
    for (g_idx, t_id, step, sc) in all_points:
        score_to_threads[sc].add(t_id)
    repeated_scores = {
        sc
        for sc, tset in score_to_threads.items() if len(tset) > 1
    }

    # For each repeated score, find earliest occurrence (lowest global_line_index)
    earliest_occurrence = {}  # sc -> (g_idx, t_id, local_step)
    for (g_idx, t_id, st, sc) in all_points:
        if sc in repeated_scores:
            if sc not in earliest_occurrence or g_idx < earliest_occurrence[
                    sc][0]:
                earliest_occurrence[sc] = (g_idx, t_id, st)

    # Build per-thread data for plotting
    thread_data = defaultdict(list)
    for (g_idx, t_id, st, sc) in all_points:
        thread_data[t_id].append((st, sc))

    for t_id in thread_data:
        thread_data[t_id].sort(key=lambda x: x[0])

    # We'll store each thread's line color
    thread_color = {}
    # We'll store final points for labeling
    final_points = {}

    plt.figure()
    ax = plt.gca()

    # Plot each thread, capturing its color
    for t_id, pts in thread_data.items():
        steps = [p[0] for p in pts]
        scores = [p[1] for p in pts]
        # Plot returns a list of line objects (usually length 1)
        line_obj, = plt.plot(steps,
                             scores,
                             linestyle='-',
                             marker=None,
                             label=f"Thread {t_id}")
        # Extract the color
        line_color = line_obj.get_color()
        thread_color[t_id] = line_color

        if pts:
            final_points[t_id] = pts[-1]  # (last_step, last_score)

    # Draw once so bounding boxes are known
    plt.draw()

    # Label earliest occurrence of repeated scores, colored by the earliest thread
    for sc, (earliest_gidx, earliest_thr,
             earliest_step) in earliest_occurrence.items():
        place_label_avoiding_collision(
            ax=ax,
            x=earliest_step,
            y=sc,
            text=f"{sc:.3f}",
            color=thread_color[
                earliest_thr],  # same color as earliest thread's line
            fontsize=10,
            ha='center',
            va='bottom')

    # Label final points for each thread, in the thread's color
    for t_id, (final_step, final_score) in final_points.items():
        place_label_avoiding_collision(
            ax=ax,
            x=final_step,
            y=final_score,
            text=f"{final_score:.3f}",
            color=thread_color[t_id],  # same color as that thread’s line
            fontsize=10,
            ha='center',
            va='bottom',
            fontweight='bold')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.xlabel("Sequential Step (per thread)")
    plt.ylabel("Best Score So Far (3 decimals)")
    plt.title("Threads: Final + Earliest Occurrence of Repeated Scores\n"
              "(Line/Label Colors Match, Simple Collision Avoidance)")
    plt.legend()
    plt.savefig(output_png)
    plt.close()


if __name__ == "__main__":
    plot_same_color_labels(filename="pytorch_6963654.out",
                           output_png="progress.png")
    print("Done! See 'progress.png' for the chart.")
