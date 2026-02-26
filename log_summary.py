import os
import re
import pandas as pd
import glob


# 定义函数用于从文件内容中提取实验结果
def extract_results(file_content):
    # 使用正则表达式提取预测长度、MSE和MAE
    pattern = r'mse:(\d+\.\d+), mae:(\d+\.\d+), rmse:(\d+\.\d+)'
    match = re.findall(pattern, file_content)

    if len(match) != 0:
        match = match[0]
        mse = float(match[0])
        mae = float(match[1])
        rse = float(match[2])
        results = (mse, mae, rse)
    else:
        results = (None, None, None)
        # results = None
    return results


# 定义函数用于从文件名中提取参数
def extract_params(filename):
    # 从文件名中提取参数部分（去掉_summary.log）
    params = os.path.basename(filename).replace('.log', '')
    return params


# 主函数
def process_log_files(folder_path):
    all_results = []

    # 查找所有以.log结尾的文件
    log_files = glob.glob(os.path.join(folder_path, '*.log'))

    for log_file in log_files:
        params = extract_params(log_file)

        # 尝试不同的编码来读取文件
        encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'gb2312', 'gb18030']
        content = None

        for encoding in encodings:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    content = f.read()
                # 如果成功读取，跳出循环
                break
            except UnicodeDecodeError:
                continue

        # 如果所有编码都失败，跳过此文件
        if content is None:
            print(f"警告: 无法读取文件 {log_file}，已跳过")
            continue

        try:
            pred_len = params.split('_pred_')[1].split('_')[0]
        except IndexError:
            print(f"警告: 无法从文件名 {log_file} 提取预测长度，已跳过")
            continue

        results = extract_results(content)


        all_results.append({
            '数据集': params.split('_',1)[1].split('_seed')[0],
            '参数':params.split('_label_')[1],
            # '参数': params.split('_fc_dropout_')[0] + params.split('head_dropout')[1] + params.split('_fc_dropout_')[1].split('head_dropout')[0],
            '预测长度': pred_len,
            'MSE': results[0],
            'MAE': results[1],
            'RSE': results[2],
        })

    if not all_results:
        print("没有找到有效的结果数据。请检查文件内容和正则表达式。")
        return

    # 创建第一个文件的DataFrame：参数，预测长度，MSE，MAE
    df1 = pd.DataFrame(all_results)

    # # 创建第二个文件的DataFrame：参数，MSE均值，MAE均值
    # df2 = df1.groupby(['数据集', '参数']).agg({'MSE': 'mean', 'MAE': 'mean', 'RSE': 'mean'}).reset_index()
    # df2.columns = ['数据集', '参数', 'MSE均值', 'MAE均值', 'RSE均值']
    df1['组合长度'] = df1.groupby(['数据集', '参数'])['MSE'].transform('size')

    # 掩码掉组合长度不等于 4 的行
    df1_masked = df1[df1['组合长度'] == 4]

    df2 = df1_masked.groupby(['数据集', '参数']).agg({'MSE': 'mean', 'MAE': 'mean', 'RSE': 'mean'}).reset_index()
    df2.columns = ['数据集', '参数', 'MSE 均值', 'MAE 均值', 'RSE 均值']

    # 保存到CSV文件
    # df1.to_csv(os.path.join(folder_path, 'detailed_results.csv'), index=False)
    # df2.to_csv(osy.path.join(folder_path, 'average_results.csv'), index=False)

    target_dir = 'baseline_logs/'

    df1.to_csv(target_dir + 'detailed_'+ folder_path.split('/')[-1] +'.csv', index=False)
    df2.to_csv(target_dir + 'average_'+ folder_path.split('/')[-1] +'.csv', index=False)

    df1.to_excel(target_dir + 'detailed_' + folder_path.split('/')[-1] + '.xlsx', index=False)
    df2.to_excel(target_dir + 'average_' + folder_path.split('/')[-1] + '.xlsx', index=False)

    print(f"已处理 {len(log_files)} 个日志文件")
    print(f"找到 {len(all_results)} 个有效结果")
    # print(f"详细结果已保存到 {os.path.join(folder_path, 'detailed_results.csv')}")
    # print(f"平均结果已保存到 {os.path.join(folder_path, 'average_results.csv')}")

def delete_trace_logs(directory):
    """
    删除指定目录中所有第一行以 "Trace" 开头的 .log 文件

    参数:
        directory (str): 需要检查和清理的目录路径
    """
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return

    # 检查目录下的所有文件
    deleted_files = []
    try:
        all_count = 0
        count = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                # 只处理.log文件
                if entry.is_file() and entry.name.lower().endswith('.log'):
                    file_path = entry.path
                    all_count += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # 判断是否包含 "Traceback"
                        has_traceback = any(line.strip().startswith("Trace") for line in lines)
                        #
                        # # 判断最后一行是否以 ">>>" 开头（仅当文件有内容时）
                        # if lines:
                        #     last_line = lines[-1].strip()
                        #     not_ends_with_prompt = not last_line.startswith("mse")
                        # else:
                        #     # 空文件也视为“无效”
                        #     not_ends_with_prompt = True
                        not_ends_with_prompt = False

                        # 如果包含 Traceback 或 最后一行不是 >>>，则标记删除
                        if has_traceback or not_ends_with_prompt:
                            deleted_files.append(file_path)

                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                        deleted_files.append(file_path)
                        count+=1

    except Exception as e:
        print(f"扫描目录 {directory} 时出错: {e}")

    # 如果找到需要删除的文件，询问用户确认
    print("共 " + str(all_count) + " 个文件, 其中 " + str(count) + " 个文件读取出错")
    if deleted_files:
        print("找到以下文件将被删除:")
        for file in deleted_files:
            print(file)

        confirm = input("是否确认删除这些文件? (y/n): ")

        if confirm.lower() == 'y':
            # 删除文件
            for file in deleted_files:
                try:
                    os.remove(file)
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除文件 {file} 时出错: {e}")
        else:
            print("操作已取消")
    else:
        print("没有找到符合条件的文件")


# 运行脚本（替换为您的文件夹路径）

# folder_path = 'baseline/gpt4mts_sort'  # _with_prior_y logs_clip_corr_prepare_new logs_clip_corr_two_stage当前文件夹，您可以修改为具体路径 logs_clip_corr_two_stage
# folder_path = 'A_result/only_ts_TimesNet'
# folder_path = 'A_result/3stage_longpred'
folder_path = 'A_result/results_stat'
# folder_path = 'A_result/healthfix1_'

# print(f"正在检查目录: {folder_path}")
delete_trace_logs(folder_path)
process_log_files(folder_path)










