################################################################################
# Copyright 2021-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import argparse
import sys
import json


def table_print(prefix, data, space):
    out = ""
    remain_func = prefix + data[0]
    if len(prefix)+7 > space[0]:
        print("Too many levels of calls")
        exit(0)
    startidx = 1
    is_md = (args.out == "md")
    if is_md:
        out = "|"
    if args.no_break_line:
        startidx = 0
    else:
        while True:
            if len(remain_func) <= space[0]:
                out += remain_func
                out += ' ' * (space[0]-len(remain_func))
                if is_md:
                    out += "|"
                break
            out += remain_func[:space[0]]
            if is_md:
                out += "<br/>"
            else:
                out += "\n"

            remain_func = prefix + "  .." + remain_func[space[0]:]

    for i in range(startidx, len(data)):
        s = str(data[i])
        assert(len(data) < space[i])
        out += s
        out += ' ' * (space[i]-len(s))
        if is_md:
            out += "|"
    return out.replace("[ ", "[&nbsp;")


class func_entry:
    def __init__(self, name):
        self.name = name
        self.subfuncs = dict()
        self.ticks = 0
        self.calls = 0
        self.prefetch_num = 0

    def get_or_create_subfunc(self, name):
        if not self.subfuncs.get(name, None):
            newfunc = func_entry(name)
            self.subfuncs[name] = newfunc
            return newfunc
        else:
            return self.subfuncs[name]

    def print(self, level, total_ticks):
        if args.out == "md":
            prefix = "[ " * level
        else:
            prefix = "| " * level

        percent = "{:.2f}".format(self.ticks/float(total_ticks)*100)
        avg = 0 if self.calls == 0 else self.ticks / self.calls / 1e3
        avg = "{:.5f}".format(avg)
        ticks = "{:.5f}".format(self.ticks/1e3)
        if self.calls:
            print(table_print(prefix, (self.name, ticks,
                                       percent, self.calls, avg), (70, 14, 8, 12, 14)))
        funcs = list(self.subfuncs.values())
        funcs.sort(key=lambda x: x.ticks, reverse=True)
        for f in funcs:
            f.print(level+1, total_ticks)

    def print_table(self):
        print(table_print("", ("Function", "Time(ms)",
                               "% Time", "Calls", "Avg ms"), (70, 14, 8, 12, 14)))
        if args.out == "md":
            print("|", "---|" * 5)
        self.print(0, self.ticks)
        if args.out == "md":
            print("\n")

    def print_csv(self, total_ticks):
        percent = self.ticks/float(total_ticks)*100
        line = "{},{},{},{}".format(self.name, self.ticks, percent, self.calls)
        print(line)
        for f in self.subfuncs:
            self.subfuncs[f].print_csv(total_ticks)


def parse_file(f, include_init):
    thread_stacks = dict()
    thread_main = dict()
    prefetch_num_by_tid = dict()

    def get_stack_by_tid(tid, tick):
        nonlocal thread_stacks, thread_main
        if tid not in thread_stacks:
            main_func = func_entry("!main")
            func_stack = [(main_func, tick)]
            thread_stacks[tid] = func_stack
            thread_main[tid] = main_func
            return func_stack
        else:
            return thread_stacks[tid]
    last_top_func = dict()
    init_top = False
    init_tick = 0
    for line in f:
        if line.startswith('{"pid":1, "tid":'):
            if line[-2:] == ',\n':
                line = line[:-2]
            jsdict = json.loads(line)

            name = jsdict["name"]
            tick = jsdict["ts"]
            tid = jsdict["tid"]
            isout = 0 if jsdict["ph"] == 'B' else 1
            func_stack = get_stack_by_tid(tid, tick)
            if isout:
                func, intick = func_stack[-1]
                if name != func.name and name == "barrier@2":
                    continue
                assert(name == func.name)
                if name == "barrier_internal@3":
                    prefetch_times = jsdict["args"]["flop"]
                    prefetch_num_by_tid[tid] = prefetch_num_by_tid.get(tid,0)+prefetch_times
                if not include_init:
                    if init_top and intick > init_tick:
                        func.ticks += (tick - intick)
                        func.calls += 1
                    elif not init_top and len(func_stack) == 2 and not name.startswith("__sc_init"):
                        init_top = True
                        init_tick = tick
                else:
                    func.ticks += (tick - intick)
                    func.calls += 1
                func_stack.pop()
                if len(func_stack) == 1:
                    last_top_func[tid] = func
            else:
                func, parenttick = func_stack[-1]
                last_top = last_top_func.get(tid, None)
                if len(func_stack) == 1 and name.startswith("barrier") and last_top and not last_top.name.startswith("barrier"):
                    sub_func = last_top.get_or_create_subfunc(name)
                else:
                    sub_func = func.get_or_create_subfunc(name)
                func_stack.append((sub_func, tick))
    for tid, main_func in thread_main.items():
        for func in main_func.subfuncs.values():
            main_func.ticks += func.ticks
        main_func.prefetch_num = prefetch_num_by_tid.get(tid,0)
    return thread_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="sctrace.json")
    parser.add_argument("--out", type=str, default="tree",
                        choices=["csv", "tree", "md"])
    parser.add_argument("--include-init", action='store_true',
                        default=False, help="Include the const-folding and initalization stage.")
    parser.add_argument("--no-break-line", action='store_true',
                        default=False, help="Don't break the function name into multiple lines")
    args = parser.parse_args()
    funcs = None
    if args.file == "":
        funcs = parse_file(sys.stdin, args.include_init)
    else:
        with open(args.file) as f:
            funcs = parse_file(f, args.include_init)
    if args.out == "tree" or args.out == "md":
        for tid, func in sorted(funcs.items(), key=lambda kv: kv[0]):
            if len(funcs) > 1:
                print("=============\nTID={}".format(tid))
            if func.prefetch_num > 0:
                print("prefetch:", func.prefetch_num)
            func.print_table()
    elif args.out == "csv":
        for tid, func in funcs.items():
            func.print_csv(func.ticks)
    else:
        print("Bad out type")
