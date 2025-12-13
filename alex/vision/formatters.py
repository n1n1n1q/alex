class VisionFormatter:

    @staticmethod
    def format_scores(scores: dict, as_percent: bool = True) -> str:

        lines = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar_len = int(score * 20) if score <= 1.0 else int(min(score / 5, 1.0) * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            if as_percent and score <= 1.0:
                lines.append(f"  {name:15} [{bar}] {score:.1%}")
            else:
                lines.append(f"  {name:15} [{bar}] {score:.2f}")
        return "\n".join(lines)

    @staticmethod
    def generate_context_string(analysis: dict) -> str:

        summary = analysis["summary"]
        threat_info = (
            summary["top_threat"] if summary["top_threat"] else "none detected"
        )
        env = analysis["environment"]
        entities = analysis["entities"]

        context = f"""=== MINECRAFT VISUAL ANALYSIS ===

QUICK SUMMARY:
  Biome: {summary['biome']} (probability: {env['biome']['probability']:.1%})
  Time: {summary['time']} (probability: {env['time']['probability']:.1%})
  Weather: {summary['weather']} (probability: {env['weather']['probability']:.1%})
  Safety: {summary['safety']} (probability: {analysis['safety']['probability']:.1%})
  Primary Threat: {threat_info}
  Top Resource: {summary['top_resource']}

DETAILED PROBABILITIES:

Environment/Biome:
{VisionFormatter.format_scores(env['biome']['probabilities'])}

Time of Day:
{VisionFormatter.format_scores(env['time']['probabilities'])}

Weather:
{VisionFormatter.format_scores(env['weather']['probabilities'])}

Safety Assessment:
{VisionFormatter.format_scores(analysis['safety']['probabilities'])}

Hostile Mobs (threat detection):
{VisionFormatter.format_scores(entities['hostile_mobs']['probabilities'])}

Passive Mobs:
{VisionFormatter.format_scores(entities['passive_mobs']['probabilities'])}

Resources:
{VisionFormatter.format_scores(analysis['resources']['probabilities'])}

Structures:
{VisionFormatter.format_scores(analysis['structures']['probabilities'])}

=== END ANALYSIS ===
"""
        return context

    @staticmethod
    def generate_combined_context(global_analysis: dict, spatial_analysis: dict) -> str:

        global_context = VisionFormatter.generate_context_string(global_analysis)
        spatial_desc = spatial_analysis["description"]

        if spatial_analysis.get("detections"):
            detections_str = "\n\nSPATIAL DETECTIONS (with locations):\n"
            for det in spatial_analysis["detections"][:5]:
                detections_str += (
                    f"  - {det['object']}: "
                    f"{det['horizontal_zone']}, {det['depth_zone']} "
                    f"(confidence: {det['confidence']:.2%})\n"
                )
            spatial_desc = spatial_desc + detections_str

        combined = f"""{global_context}

{spatial_desc}
"""
        return combined
