"""V-Attack configuration."""


def resolve_cli_kwargs(attack_name, args, context):
    return {"target_text": getattr(args, "target_text", "") or ""}
