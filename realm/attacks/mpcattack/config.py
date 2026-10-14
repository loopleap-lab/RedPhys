def resolve_cli_kwargs(attack_name, args, context):
    return {"source_images_dir": context["source_dir"], "target_images_dir": context["target_dir"]}
