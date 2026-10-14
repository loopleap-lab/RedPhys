def resolve_cli_kwargs(attack_name, args, context):
    if attack_name == "advedm":
        return {"target_images_dir": context.get("target_dir", "")}
    return {"target_text": getattr(args, "target_text", "")}
