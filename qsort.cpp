#include <iostream>
#include <vector>
#include <stack>

int partion(std::vector<int>& vec, int start, int end)
{
    int i = start, j = end;
    int compare_num = vec[start];
    int pivot; // pivot左侧都小于pivot, 右侧都大于pivot
    while (i < j) {
        while (i < j && compare_num <= vec[j]) j--; 
        while (i < j && compare_num >= vec[i]) i++;
        std::swap(vec[i], vec[j]);
    }
    std::swap(vec[start], vec[i]);
    return i;
}

void qsort(std::vector<int>& vec, int start, int end) 
{
    if (start >= end) return;
    int pivot = partion(vec, start, end);
    qsort(vec, start, pivot - 1);
    qsort(vec, pivot + 1, end);
}

void qsort_stack(std::vector<int>& vec, int start, int end)
{
    std::stack<std::pair<int, int>> s;
    s.push(std::pair<int, int> {start, end});
    while (!s.empty()) {
        auto& record = s.top();
        s.pop();
        int st = record.first;
        int en = record.second;
        if (st >= en) continue;
        int pivot = partion(vec, st, en);
        s.push(std::pair<int, int> {st, pivot - 1});
        s.push(std::pair<int, int> {pivot + 1, en}); 
    }
}

int main()
{
    std::vector<int> vec{7, 6, 5, 3, 4, 3, 2, 1};
    qsort_stack(vec, 0, vec.size() - 1);
    // qsort(vec, 0, vec.size() - 1);
    for (auto &num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}