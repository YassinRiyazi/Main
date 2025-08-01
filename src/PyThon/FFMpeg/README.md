
def img_mkr_rotation(experiment, N, angle):
    subprocess.run([
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", f"./{experiment}/{experiment.split('/')[-1]}.mp4",
                "-vf", f"crop=1280:ih-{N}:0:0,rotate={angle}*(PI/180):ow=rotw({angle}*(PI/180)):oh=roth({angle}*(PI/180)):c=white,select='lt(n\\,15)'",
                "-vsync", "vfr",
                "-q:v", "2",
                f"{experiment}/%06d.jpg"
    ])