import os
import argparse
import subprocess
import re
import pandas as pd
import datetime
import sys # Added for error output
from typing import Tuple, Iterator

# --- Copied simple_fasta_parser from Get_data.py ---
def simple_fasta_parser(fasta_filename: str) -> Iterator[Tuple[str, str, str]]:
    """
    A simple generator function to parse FASTA files without Biopython.
    Yields tuples of (id, description, sequence).
    Handles potential file errors.
    """
    sequence = ''
    header = None
    seq_id = None
    description = None

    try:
        with open(fasta_filename, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        # Ensure sequence is not empty before yielding
                        if sequence:
                            yield (seq_id, description, sequence)
                        else:
                             print(f"Warning: Empty sequence for header '{header}' in {fasta_filename}", file=sys.stderr)

                    header = line[1:]
                    parts = header.split(None, 1) # Split only on the first whitespace
                    seq_id = parts[0]
                    description = header # Keep the full header as description
                    sequence = ''
                elif header is not None: # Only append if we are inside a sequence block
                    # Basic check for sequence characters (can be expanded)
                    if all(c in 'ACGTNRYWSMKBDHV-' for c in line.upper()):
                        sequence += line
                    else:
                         print(f"Warning: Non-standard character found in sequence for header '{header}' in {fasta_filename}. Line: '{line}'", file=sys.stderr)
                         # Decide whether to skip the line or the whole sequence

            # Yield the last sequence in the file
            if header is not None and sequence:
                yield (seq_id, description, sequence)
            elif header is not None and not sequence:
                 print(f"Warning: Empty sequence for last header '{header}' in {fasta_filename}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_filename}", file=sys.stderr)
        # Raise the error or return an empty iterator depending on desired behavior
        raise # Re-raise the error to stop the process
    except Exception as e:
        print(f"Error parsing FASTA file {fasta_filename}: {e}", file=sys.stderr)
        raise # Re-raise the error

# --- End of copied function ---


def extract_fasta_header_info(header):
    """从标题中提取ID、日期、位置和分支信息
    标题格式: "PV403655.1 |2022-07-14|USA|B.1" or just "PV403655.1"
    """
    # Use the full header string passed from simple_fasta_parser
    parts = header.split('|')

    # 提取ID (清除可能的空格) - Use the first part before any whitespace as ID if no '|'
    id_part = header.split(None, 1)[0].strip() # More robust ID extraction

    # 提取日期、位置和分支 based on '|' separator
    date = parts[1].strip() if len(parts) > 1 else ""
    location = parts[2].strip() if len(parts) > 2 else "Unknown"
    clade = parts[3].strip() if len(parts) > 3 else "Unknown"

    # Attempt to extract date using regex if not found via '|'
    if not date:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', header)
        if date_match:
            date = date_match.group(1)

    return {
        'ID': id_part,
        'Date': date,
        'Location': location,
        'Clade': clade
    }

def calculate_n_content(sequence):
    """计算序列中N碱基的含量和比率"""
    if not sequence: # Handle empty sequence case
        return 0, 0.0
    n_count = sequence.upper().count('N')
    n_ratio = n_count / len(sequence)
    return n_count, n_ratio

# ... (之前的代码保持不变) ...

def filter_strains(fasta_file_path, max_n_ratio, output_fasta_template, output_report, min_size=0):
    """过滤N含量高于阈值的序列 (不使用Biopython)"""
    print(f"1. 筛选N比率低于{max_n_ratio}且长度 >= {min_size} 的序列...")

    # 初始化列表
    filtered_sequences_data = [] # Store tuples (id, sequence)
    quality_report = []
    metadata_records = []
    all_valid_dates = [] # Store datetime objects for min/max calculation

    # 用于记录日期最早的序列
    earliest_date_obj = None
    earliest_seq_id = None

    # 读取FASTA文件 using the simple parser
    try:
        for seq_id, description, sequence in simple_fasta_parser(fasta_file_path):
            if not seq_id or not sequence: # Skip if parser yielded invalid data
                continue

            # 长度过滤
            if len(sequence) < min_size:
                continue

            # 计算N含量和比率
            n_count, n_ratio = calculate_n_content(sequence)
            if n_ratio > max_n_ratio:
                continue

            # 解析标题信息
            header_info = extract_fasta_header_info(description) # Use the full description
            actual_id = seq_id

            # 记录元数据 (using actual_id)
            header_info['ID'] = actual_id # Ensure metadata uses the correct ID
            metadata_records.append(header_info)

            # 提取日期并找出最早的序列
            current_date_obj = None
            date_str = header_info.get('Date') # Get date string safely

            if date_str:
                try:
                    # First, try parsing the full date format '%Y-%m-%d'
                    current_date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    try:
                        # Second, try parsing year-month format '%Y-%m'
                        # Append '-01' to represent the first day of the month
                        current_date_obj = datetime.datetime.strptime(date_str + '-01', '%Y-%m-%d')
                        # print(f"Info: Using default day '01' for partial date '{date_str}' for sequence {actual_id}", file=sys.stderr)
                    except ValueError:
                        try:
                            # Third, try parsing year-only format '%Y'
                            # Append '-01-01' to represent January 1st
                            current_date_obj = datetime.datetime.strptime(date_str + '-01-01', '%Y-%m-%d')
                            # print(f"Info: Using default month/day '01-01' for partial date '{date_str}' for sequence {actual_id}", file=sys.stderr)
                        except (ValueError, TypeError) as date_err:
                            # If all formats fail, log the warning
                            print(f"Warning: Could not parse date '{date_str}' for sequence {actual_id} using formats %Y-%m-%d, %Y-%m, or %Y: {date_err}", file=sys.stderr)
                            pass # Continue processing even if date is invalid

                # If a date object was successfully created, process it
                if current_date_obj:
                    all_valid_dates.append(current_date_obj) # Store datetime object
                    if earliest_date_obj is None or current_date_obj < earliest_date_obj:
                        earliest_date_obj = current_date_obj
                        earliest_seq_id = actual_id

            # 添加序列和信息 (using actual_id)
            filtered_sequences_data.append((actual_id, sequence))
            quality_report.append((actual_id, len(sequence), n_count, n_ratio))

    except Exception as parse_err:
         print(f"Error during FASTA parsing or processing: {parse_err}", file=sys.stderr)
         return None, None, None, pd.DataFrame() # Indicate failure

    # ... (函数其余部分保持不变) ...

    # 按序列ID排序 (using actual_id from the tuple)
    filtered_sequences_data.sort(key=lambda x: x[0])

    # 获取日期范围 from stored datetime objects
    date_range = None
    if all_valid_dates:
        try:
            start_date = min(all_valid_dates).strftime('%Y_%m_%d')
            end_date = max(all_valid_dates).strftime('%Y_%m_%d')
            date_range = f"{start_date}-{end_date}"
        except Exception as e:
             print(f"Warning: Could not determine date range from collected dates: {e}", file=sys.stderr)

    # 添加日期范围和序列数量到输出文件名
    sequence_count = len(filtered_sequences_data)
    output_fasta = output_fasta_template.replace('.fasta', f'_{sequence_count}.fasta')
    if date_range:
        # Ensure .fasta is replaced correctly even if date_range was added before
        output_fasta = output_fasta.replace(f'_{sequence_count}.fasta', f'_{sequence_count}_{date_range}.fasta')


    # 保存筛选后的序列到FASTA文件
    try:
        with open(output_fasta, 'w') as f:
            for seq_id_out, sequence_out in filtered_sequences_data:
                f.write(f">{seq_id_out}\n{sequence_out}\n")
    except IOError as e:
        print(f"Error writing filtered FASTA file {output_fasta}: {e}", file=sys.stderr)
        return None, earliest_seq_id, date_range, pd.DataFrame(metadata_records) # Return what we have

    # 生成质量报告CSV
    try:
        df_quality = pd.DataFrame(quality_report, columns=["Accession", "Sequence_Length", "N_Count", "N_Ratio"])
        csv_report = output_report
        df_quality.to_csv(csv_report, index=False)
    except Exception as e:
        print(f"Error writing quality report CSV {csv_report}: {e}", file=sys.stderr)


    # 创建元数据数据框
    metadata_df = pd.DataFrame(metadata_records)

    print(f"筛选后的序列已保存至: {output_fasta}")
    print(f"质量报告已保存至: {csv_report}")
    print(f"总共筛选出: {sequence_count} 条序列")

    return output_fasta, earliest_seq_id, date_range, metadata_df

# ... (其余代码保持不变) ...

def prepare_processed_data(mutations_csv_path, metadata_df, output_csv_path):
    """处理突变数据与元数据的匹配"""
    print(f"4. 处理突变数据和元数据信息...")

    # 读取突变数据
    try:
        # Try reading with common issues in mind
        try:
            mutations_df = pd.read_csv(mutations_csv_path)
        except pd.errors.ParserError:
             print(f"Warning: ParserError reading {mutations_csv_path}, trying with quoting=3", file=sys.stderr)
             mutations_df = pd.read_csv(mutations_csv_path, quoting=3) # csv.QUOTE_NONE
        except FileNotFoundError:
             print(f"错误: 突变文件未找到: {mutations_csv_path}", file=sys.stderr)
             return pd.DataFrame(), None
    except Exception as e:
        print(f"读取突变文件错误: {e}", file=sys.stderr)
        return pd.DataFrame(), None

    # Check if required columns exist
    if 'Query_ID' not in mutations_df.columns or 'Mutations' not in mutations_df.columns:
         print(f"错误: 突变文件 {mutations_csv_path} 缺少 'Query_ID' 或 'Mutations' 列。", file=sys.stderr)
         return pd.DataFrame(), None

    # 将metadata_df的ID列设为索引方便查询
    if 'ID' not in metadata_df.columns:
         print(f"错误: 元数据 DataFrame 缺少 'ID' 列。", file=sys.stderr)
         return pd.DataFrame(), None
    try:
        # Ensure ID column is string and handle potential duplicates before setting index
        metadata_df['ID'] = metadata_df['ID'].astype(str).str.strip()
        metadata_indexed = metadata_df.drop_duplicates(subset=['ID']).set_index('ID')
    except Exception as e:
         print(f"设置元数据索引时出错: {e}", file=sys.stderr)
         return pd.DataFrame(), None


    # 存储最终数据的列表
    data = []

    # 处理每一行突变数据
    for _, row in mutations_df.iterrows():
        query_id_raw = row['Query_ID']
        mutations = row['Mutations'] if not pd.isna(row['Mutations']) else '' # Handle NaN mutations

        # 提取ID - More robustly handle cases where ID might be the whole string or part
        if isinstance(query_id_raw, str):
             acc_id = query_id_raw.split('|')[0].split(None, 1)[0].strip() # Try splitting by | and whitespace
        else:
             acc_id = str(query_id_raw).strip() # Fallback if not string

        # 如果在元数据中找到匹配项，创建数据条目
        if acc_id in metadata_indexed.index:
            metadata = metadata_indexed.loc[acc_id]

            # Ensure metadata columns exist before accessing
            date_val = metadata.get('Date', '')
            location_val = metadata.get('Location', 'Unknown')
            clade_val = metadata.get('Clade', 'Unknown')

            data_entry = {
                'ID': acc_id,
                'Date': date_val,
                'Location': location_val,
                'Clade': clade_val,
                'Mutations_str': mutations
            }

            data.append(data_entry)
        # else: # Optional: Log IDs not found in metadata
             # print(f"Debug: ID '{acc_id}' from mutations file not found in metadata index.")


    # 创建数据框
    if not data:
         print("警告: 未找到匹配记录，将生成空的输出文件。")
         # Still create an empty DataFrame with correct columns for consistency
         result_df = pd.DataFrame(columns=['ID', 'Date', 'Location', 'Clade', 'Mutations_str'])
    else:
         result_df = pd.DataFrame(data)

    # 转换日期列
    def parse_date(date_str):
        if pd.isna(date_str) or str(date_str).strip() == '':
            return pd.NaT
        try:
            # Try common formats
            return pd.to_datetime(date_str, errors='coerce') # Coerce invalid dates to NaT
        except Exception as e:
            # print(f"警告: 无法解析日期 '{date_str}': {e}", file=sys.stderr) # Reduce verbosity
            return pd.NaT

    result_df['Date'] = result_df['Date'].apply(parse_date)

    # 获取日期范围
    date_range = None
    try:
        valid_dates = result_df['Date'].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().strftime('%Y_%m_%d')
            max_date = valid_dates.max().strftime('%Y_%m_%d')
            date_range = f"{min_date}-{max_date}"
    except Exception as e:
        print(f"警告: 无法确定日期范围: {e}", file=sys.stderr)

    # 保存数据框到CSV文件（使用制表符分隔）
    try:
        result_df.to_csv(output_csv_path, sep='\t', header=True, index=False) # Changed index=False
        print(f"处理后的数据已保存至: {output_csv_path}")
    except Exception as e:
         print(f"错误: 无法写入最终处理数据到 {output_csv_path}: {e}", file=sys.stderr)
         # Decide if failure to write should prevent returning the DataFrame

    return result_df, date_range


def run_pipeline(input_file, output_dir, ratio=0.001, ref=None, n=4, min_size=0):
    """执行完整的处理流程"""
    # 确保输出目录存在
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建输出目录 {output_dir}: {e}", file=sys.stderr)
        return None # Critical error

    print(f"处理FASTA文件: {input_file}")

    # 设置输出文件路径 (use os.path.splitext for robustness)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    filtered_fasta_template = os.path.join(output_dir, f"{base_name}_filtered_N_lt_{ratio}.fasta")
    quality_report = os.path.join(output_dir, f"{base_name}_quality_report_{ratio}.csv")
    processed_data_base = os.path.join(output_dir, f"{base_name}_processed_data")
    halign_output_base = os.path.join(output_dir, f"{base_name}_halign4")
    mutations_csv = os.path.join(output_dir, f"{base_name}_mutations_result.csv") # Expected output from variant_mark_ljj.py

    # 1. 执行filter_strains函数
    filter_result = filter_strains(
        input_file, ratio, filtered_fasta_template, quality_report, min_size
    )
    # Check if filter_strains indicated failure
    if filter_result is None or filter_result[0] is None:
         print("错误: filter_strains 失败或未生成过滤后的 FASTA 文件。流程中止。", file=sys.stderr)
         return None
    filtered_fasta, earliest_seq_id, date_range, metadata_df = filter_result

    # Check if the filtered FASTA file actually exists
    if not os.path.exists(filtered_fasta):
         print(f"错误: 过滤后的 FASTA 文件未找到: {filtered_fasta}。流程中止。", file=sys.stderr)
         return None

    # 如果未指定参考序列，使用日期最早的序列作为参考
    if ref is None:
        if earliest_seq_id:
            ref = earliest_seq_id
            print(f"使用日期最早的序列作为参考: {ref}")
        else:
            print("错误: 未指定参考序列且无法找到最早序列。流程中止。", file=sys.stderr)
            return None # Reference is needed for variant marking

    # 设置带有日期范围的halign输出文件名
    halign_output = f"{halign_output_base}.fasta" # Default name
    if date_range:
        halign_output = f"{halign_output_base}_{date_range}.fasta"

    # 2. 执行halign4命令
    print(f"2. 执行多序列比对(halign4)...")
    # Ensure paths with spaces are quoted if necessary, though os.path.join handles separators
    halign_cmd = f"halign4 \"{filtered_fasta}\" \"{halign_output}\" -t {n}"
    try:
        print(f"运行命令: {halign_cmd}")
        subprocess.run(halign_cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"halign4比对结果已保存到: {halign_output}")
    except subprocess.CalledProcessError as e:
        print(f"错误: 执行 halign4 失败: {e}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return None # Critical error
    except FileNotFoundError:
         print("错误: 'halign4' 命令未找到。请确保已安装并在系统 PATH 中。", file=sys.stderr)
         return None

    # 3. 执行variant_mark_ljj Python命令
    print(f"3. 标记变异位点...")
    variant_script = "variant_mark_ljj.py" # Assume it's in PATH or same dir
    # Pass base_name to the script if it uses it for output naming
    variant_cmd = f"python {variant_script} -fas \"{halign_output}\" -ref \"{ref}\" -o \"{output_dir}\" -t {n} --base_name \"{base_name}\""
    try:
        print(f"运行命令: {variant_cmd}")
        subprocess.run(variant_cmd, shell=True, check=True, capture_output=True, text=True)
        print("变异标记完成。")
        # Verify expected output file exists
        if not os.path.exists(mutations_csv):
             print(f"警告: 预期的突变输出文件未找到: {mutations_csv}", file=sys.stderr)
             # Decide if this is critical. If prepare_processed_data needs it, it is.
             print("错误: 缺少突变文件，无法继续。流程中止。", file=sys.stderr)
             return None
    except subprocess.CalledProcessError as e:
        print(f"错误: 执行变异标记脚本失败: {e}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return None # Critical error
    except FileNotFoundError:
         print(f"错误: 变异标记脚本 '{variant_script}' 未找到。", file=sys.stderr)
         return None

    # 4. 处理突变数据与元数据匹配
    final_output_path_base = f"{processed_data_base}.csv"
    df, meta_date_range = prepare_processed_data(
        mutations_csv_path=mutations_csv,
        metadata_df=metadata_df,
        output_csv_path=final_output_path_base # Save to base name first
    )

    # Check if prepare_processed_data returned a DataFrame
    if df is None:
         print("错误: 处理突变和元数据失败。流程中止。", file=sys.stderr)
         return None

    # 使用从数据中获取的日期范围
    final_date_range = meta_date_range # Use date range from the final merged data

    # 重命名文件以包含日期范围
    final_output_path = final_output_path_base # Default path
    if final_date_range:
        final_output_path_with_range = f"{processed_data_base}_{final_date_range}.csv"
        try:
            # Check if the base file exists before renaming
            if os.path.exists(final_output_path_base):
                 os.rename(final_output_path_base, final_output_path_with_range)
                 final_output_path = final_output_path_with_range # Update path
                 print(f"最终处理数据已重命名为包含日期范围: {final_output_path}")
            else:
                 # This case might happen if prepare_processed_data failed to write the file
                 print(f"警告: 基础输出文件未找到，无法重命名: {final_output_path_base}", file=sys.stderr)
                 # Keep final_output_path as the base path, even though it might not exist
        except Exception as e:
            print(f"警告: 无法重命名带有日期范围的文件: {e}", file=sys.stderr)
            # Keep final_output_path as the base path
    else:
        # If no date range, the file should already exist with the base name
        if os.path.exists(final_output_path_base):
             print(f"最终处理数据已保存至: {final_output_path_base}")
        else:
             # This indicates prepare_processed_data might have failed to write
             print(f"警告: 最终输出文件未找到: {final_output_path_base}", file=sys.stderr)


    print("数据处理流程成功完成!")
    # Return the DataFrame and the final path (which might or might not exist if writing failed)
    return df


def main():
    parser = argparse.ArgumentParser(description='序列分析流程')
    parser.add_argument('--input_file', required=True, help='输入FASTA文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--ratio', type=float, default=0.001, help='最大N碱基比率，默认0.001')
    parser.add_argument('--min_size', type=int, default=0, help='最小序列长度，默认0(不过滤)')
    parser.add_argument('--ref', default=None, help='参考序列ID(默认:最早日期的序列)')
    parser.add_argument('--n', type=int, default=4, help='线程数，默认4')

    args = parser.parse_args()

    # 运行完整流程
    result_df = run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        ratio=args.ratio,
        ref=args.ref,
        n=args.n,
        min_size=args.min_size
    )

    # Check if run_pipeline returned a DataFrame (indicating success)
    if result_df is not None:
         print("主程序: 流程成功结束。")
         sys.exit(0)
    else:
         print("主程序: 流程因错误中止。", file=sys.stderr)
         sys.exit(1)


if __name__ == "__main__":
    main()