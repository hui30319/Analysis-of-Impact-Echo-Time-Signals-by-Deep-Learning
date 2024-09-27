import os
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
current_dir = os.path.dirname(os.path.realpath(__file__))

class DynFileGenerator:
    def __init__(self, config):
        self.x_size = config["x_size"]
        self.y_size = config["y_size"]
        self.x_start = config["x_start_end"][0]
        self.y_start = config["y_start_end"][0]
        self.x_end = config["x_start_end"][1]
        self.y_end = config["y_start_end"][1]
        self.step = config["step"]
        self.bcs = config["bcs"]
        self.cracks = config["cracks"]
        self.depths = config["depths"]
        self.balls = config["balls"]
        self.batch_size = config["batch_size"]
        self.load_dir_dyn = config["load_dir_dyn"]
        self.save_dir_dyn = config["save_dir_dyn"]
        self.save_dir_bat = config["save_dir_bat"]
        self.save_dir_cmd = config["save_dir_cmd"]
        self.out_points = config["out_points"]

    def change_str(self, string, idx, new_str):
        str2list = list(string)
        str2list[idx:] = new_str
        return ''.join(str2list)

    def find_word(self, txt):
        db_word = "DATABASE HISTORY CARDS"
        load_word = "LOAD SEGMENT CARDS"
        curve_start_word = "LOAD CURVE CARDS"
        curve_end_word = "NODE INFORMATION"
        db_lines = load_lines = curve_start = curve_end = None
        with open(txt, 'r') as f:
            for i, lines in enumerate(f):
                if db_word in lines:
                    db_lines = i + 6
                if load_word in lines:
                    load_lines = i + 6
                if curve_start_word in lines:
                    curve_start = i + 8
                if curve_end_word in lines:
                    curve_end = i - 3
                    break
        return db_lines, load_lines, curve_start, curve_end

    def create_dyn_files(self):
        logging.info("Starting to generate .dyn files...")
        for bc in self.bcs:
            boundary_file = os.path.join(current_dir, f"./raw/boundary/{bc}.txt")
            if not os.path.exists(boundary_file):
                logging.error(f"Boundary condition file not found: {boundary_file}")
                continue
            
            with open(boundary_file) as f:
                boundary_content = f.read()

            for crack in self.cracks:
                for depth in self.depths:
                    for ball in self.balls:
                        load_file = os.path.join(current_dir, f"./raw/load/{ball:02d}mm.txt")
                        if not os.path.exists(load_file):
                            logging.error(f"Steel ball file not found: {load_file}")
                            continue
                        
                        with open(load_file) as f:
                            load_content = f.read()

                        root_dir = os.path.join(f"{self.load_dir_dyn}/", f"{crack}/{bc}/{depth:02d}cm/{ball:02d}mm")
                        os.makedirs(root_dir, exist_ok=True)

                        fname = f"{self.load_dir_dyn}/{crack}/{depth:02d}cm.dyn"
                        db, load, curve_start, curve_end = self.find_word(fname)
                        if db is None or load is None:
                            logging.error(f"dyn. file not found: {fname}")
                            continue

                        with open(fname, 'r') as f:
                            lines = f.readlines()

                        lines[load - 9] = self.change_str(lines[load - 9], -1, f"\n{boundary_content}\n")
                        lines[curve_start: curve_end + 1] = ""
                        lines[curve_start - 1] = self.change_str(lines[curve_start - 1], -1, f"\n{load_content}\n")

                        for x in tqdm(range(self.x_start, self.x_end + 1, self.step)):
                            for y in range(self.y_start, self.y_end + 1, self.step):
                                local_dir = f"{root_dir}/x{x:02d}y{y:02d}"
                                os.makedirs(local_dir, exist_ok=True)

                                lines[db] = self.change_str(lines[db], 0, 
                                                            f"{1 + x + (self.x_size + 1) * (y - 4):10d}"\
                                                            f"{2 + x + (self.x_size + 1) * (y - 4):10d}"\
                                                            f"{2 + x + (self.x_size + 1) * y + 4:10d}"\
                                                            f"{(self.x_size + 3) + x + (self.x_size + 1) * y + 4:10d}"\
                                                            f"{(self.x_size + 3) + x + (self.x_size + 1) * (y + 4):10d}"\
                                                            f"{(self.x_size + 2) + x + (self.x_size + 1) * (y + 4):10d}"\
                                                            f"{(self.x_size + 2) + x + (self.x_size + 1) * y - 4:10d}"\
                                                            f"{1 + x + (self.x_size + 1) * y - 4:10d}\n")

                                lines[load] = self.change_str(lines[load], 30, 
                                                              f"{1 + x + (self.x_size + 1) * y:10d}"\
                                                              f"{2 + x + (self.x_size + 1) * y:10d}"\
                                                              f"{(self.x_size + 3) + x + (self.x_size + 1) * y:10d}"\
                                                              f"{(self.x_size + 2) + x + (self.x_size + 1) * y:10d}\n")

                                with open(f"{local_dir}/x{x:02d}y{y:02d}.dyn", 'w') as f:
                                    f.writelines(lines)
        logging.info(".dyn files generation complete.")

    def create_bat_files(self):
        logging.info("Starting to generate .bat files....")
        batch = 0
        num = 0
        for bc in tqdm(self.bcs):
            for crack in self.cracks:
                for depth in self.depths:
                    for ball in self.balls:
                        root_dir = os.path.join(f"{self.load_dir_dyn}/", f"{crack}/{bc}/{depth:02d}cm/{ball:02d}mm")
                        os.makedirs(self.save_dir_bat, exist_ok=True)

                        for x in range(self.x_start, self.x_end + 1, self.step):
                            for y in range(self.y_start, self.y_end + 1, self.step):
                                if not (num % self.batch_size):
                                    batch += 1
                                    num = 0

                                local_dir = f"{root_dir}/x{x:02d}y{y:02d}"
                                with open(f"{self.save_dir_bat}/{self.x_size}_{bc}_{batch:02d}.bat", 'a') as f:
                                    f.writelines(f"cd {local_dir}\n")
                                    f.writelines(f"\"C:/Program Files/ANSYS Inc/v182/ansys/bin/winx64/lsdyna_sp.exe\" I=x{x:02d}y{y:02d}.dyn\n")

                                num += 1
        logging.info(".bat files generation complete.")

    def create_cmd_files(self):
        logging.info("Starting to generate .cmd files...")
        for bc in tqdm(self.bcs):
            for crack in self.cracks:
                for depth in self.depths:
                    for ball in self.balls:
                        root_dir = os.path.join(f"{self.load_dir_dyn}/", f"{crack}/{bc}/{depth:02d}cm/{ball:02d}mm")
                        # "/".join(["../processed"] + root_dir.split("/")[1:])
                        sig_dir = os.path.join(f"{self.save_dir_dyn}/", f"{crack}/{bc}/{depth:02d}cm/{ball:02d}mm")
                        #f"{self.save_dir_dyn}"#"/".join(["../processed"] + root_dir.split("/")[-6:])
                        os.makedirs(self.save_dir_cmd, exist_ok=True)
                        os.makedirs(sig_dir, exist_ok=True)

                        for x in range(self.x_start, self.x_end + 1, self.step):
                            for y in range(self.y_start, self.y_end + 1, self.step):
                                node = [1 + x + (self.x_size + 1) * (y - 4),
                                        2 + x + (self.x_size + 1) * (y - 4),
                                        2 + x + (self.x_size + 1) * y + 4,
                                        (self.x_size + 3) + x + (self.x_size + 1) * y + 4,
                                        (self.x_size + 3) + x + (self.x_size + 1) * (y + 4),
                                        (self.x_size + 2) + x + (self.x_size + 1) * (y + 4),
                                        (self.x_size + 2) + x + (self.x_size + 1) * y - 4,
                                        1 + x + (self.x_size + 1) * y - 4]

                                local_dir = os.path.join(f"{root_dir}/", f"x{x:02d}y{y:02d}")
                                with open(f"{self.save_dir_cmd}/{bc}_{crack}.cmd", 'a') as f:
                                    f.writelines(f"ascii nodout open \"{local_dir}/nodout\"\n")
                                    for i in range(self.out_points):
                                        f.writelines(f"ascii nodout plot 3 {node[i]}\n")
                                        f.writelines(f"xyplot 1 savefile xypair \"{sig_dir}/x{x:02d}y{y:02d}--{i:05d}.txt\" 1 all\n")
                                    f.writelines(f"*\n")

        logging.info(".cmd files generation complete.")

    def remove_dyn_files(self):
        logging.info("Starting to remove .dyn files...")
        for path, subdirs, files in os.walk(self.load_dir_dyn):
            for name in files:
                if not (name.endswith(".dyn") or name.endswith("nodout") or name.endswith(".txt") or name.endswith(".ipynb") or name.endswith(".py")): #(fnmatch(name, pattern_1) or fnmatch(name, pattern_2)):
                    print(os.path.join(path, name))
                    os.remove(os.path.join(path, name))
        logging.info(".dyn files remove complete.")

config = {
    "x_size": 80,
    "y_size": 80,
    "x_start_end": [32, 48],
    "y_start_end": [32, 48],
    "step": 4,
    "bcs": ["HS", "FP"],
    "cracks": ["32x32"],
    "depths": [12],
    "balls": [4, 6, 10],
    "batch_size": 25,
    "load_dir_dyn": os.path.join(f"{current_dir}/", "./raw/dyn/3998/80x80x20"),
    "save_dir_dyn": os.path.join(f"{current_dir}/", "./processed/3998/80x80x20"),
    "save_dir_bat": os.path.join(f"{current_dir}/", "./raw/bat"),
    "save_dir_cmd": os.path.join(f"{current_dir}/", "./raw/cmd"),
    "out_points": 8
}

generator = DynFileGenerator(config)

print(current_dir)
# generator.create_dyn_files()
# generator.create_bat_files()
generator.create_cmd_files()
# generator.remove_dyn_files()